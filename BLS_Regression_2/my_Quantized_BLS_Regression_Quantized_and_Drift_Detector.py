import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error as RMSE_func
from sklearn.metrics import mean_absolute_percentage_error as MAPE_func

def percentile_clip_bounds(X, lower_percentile=0.5, upper_percentile=99.5):
    """
    
    Calculates robust clip bounds based on percentiles.
    
    """
    clip_min = np.percentile(X, lower_percentile, axis=0)
    clip_max = np.percentile(X, upper_percentile, axis=0)
    return clip_min, clip_max

def compute_quant_params(x_min, x_max, signed = False, n_bits = 8): 
    """
    
    Computes Zero-point and Scale parameters for a array (minimum and maximum of the array as the input)
    
    """
    if signed: #8bit signed -> -127 - +127 
        q_max = 2**(n_bits - 1) - 1
        q_min = -q_max

        max_abs = np.maximum(np.abs(x_min), np.abs(x_max))
        max_abs = np.where(max_abs < 1e-9, 1e-9, max_abs)
        scale = max_abs / q_max

        zero_point = np.zeros_like(x_min, dtype=np.int32) #zero-point is fix on the 0

    else: #8bit unsigned -> 0 - 255 
        q_max = 2**n_bits - 1
        q_min = 0
        
        scale = (x_max - x_min) / (q_max - q_min + 1e-9)

        zero_point = q_min - np.round(x_min / scale)  #the ceiling int
        # zero_point = np.clip(zero_point, q_min, q_max).astype(np.int32)

    scale = np.where(scale < 1e-9, 1e-9, scale) #for not dividing on zero later

    return scale, zero_point

def fake_quant(A_float, S, Zp, signed = False, n_bits = 8):
    """
    
    Fake quantization:
    first makes an elemnt quantized with the zero-point and scale, then makes it dequantized so the element gets quantization error with itself
    
    """
    if signed:
        q_max = 2**(n_bits - 1 ) - 1
        q_min = -q_max
    else:
        q_max = 2**n_bits -1
        q_min = 0
    
    A_quant_temp = np.round(A_float / S + Zp)
    A_quant = np.clip(A_quant_temp, q_min, q_max)  #now it's integer

    A_dequant = S * (A_quant - Zp) # now it's back to float, but it's noisy
    return A_dequant

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu6(x):
    return np.clip(x, 0, 6)

def pseudoinverse(A, lambd):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_reg = S / (S**2 + lambd)
    A_pinv = Vt.T @ np.diag(S_reg) @ U.T
    return A_pinv

def soft_thresholding(a, shrink):
    return np.sign(a) * np.maximum((abs(a) - shrink), 0)

def ADMM(A, y, shrink_coef = 0.001, iter = 50):
    
    n_features = A.shape[1]
    n_outputs = y.shape[1]
    W_k = np.zeros(shape=(n_features,n_outputs), dtype='double')
    O_k = np.zeros(shape=(n_features,n_outputs), dtype='double')
    U_k = np.zeros(shape=(n_features,n_outputs), dtype='double')

    part_0 = np.linalg.inv(A.T.dot(A) + np.eye(n_features))
    part_1 = part_0.dot(A.T).dot(y)

    for k in range(iter):
        W_k = part_1 + part_0.dot(O_k - U_k)                     # W(k+1)
        O_k = soft_thresholding(W_k + U_k, shrink_coef)          # O(k+1)
        U_k += W_k - O_k                                         # U(k+1)
        W_k = O_k
    
    return W_k

class BLS_regression:
    '''
    (self, lambd, shrink, n_features, n_ftr_groups, n_enhancements, n_enh_groups, buffer_size=30, activation ='sigmoid')
    lambda:         Regularization coefficient
    shrink:         Shrinkage coefficient, higher means more zero weights in feature weights "We" (good for prunning)
    n_features:     Number of feature nodes inside every feature group (Zi)
    n_ftr_groups:   Number of feature groups (n in #Zn)
    n_enhancements: Number of enhancement nodes inside every enhancement group (Hi)
    n_enh_groups:   Number of enhancement groups (n in #Hn)
    buffer_size:    Number of samples inside a buffer (for the data drift detection)
    activation:     The activation function used for creating enhancement features ('sigmoid'|'relu6')

    '''
    def __init__(self, lambd, shrink, n_features, n_ftr_groups, n_enhancements, n_enh_groups, buffer_size=30, activation = 'relu6'):
        self.shrink = shrink
        self.lambd = lambd
        self.n_features = n_features
        self.n_ftr_groups = n_ftr_groups
        self.n_enhancements = n_enhancements
        self.n_enh_groups = n_enh_groups
        self.limit = int(np.ceil(np.sqrt(np.log(0.025) * -1 * buffer_size)))
        self.activation = relu6 if activation == 'relu6' else sigmoid
        self.scaler = StandardScaler()
        self.clip_min = None
        self.clip_max = None

    def fit(self, X, y, QAT = False, per_tensor_quant=False):
        #start timing
        time_start = time.time()
        
        #number of all feature nodes
        n_Zn = self.n_features * self.n_ftr_groups
        
        #generate weights for feature nodes (We)
        We = []
        for i in range(self.n_ftr_groups):
            np.random.seed(i)
            We.append(2 * np.random.randn(X.shape[1]+1,self.n_features) - 1)
        
        #generate weights for enhancement nodes (Wh)
        Wh = []
        for j in range(self.n_enh_groups):
            np.random.seed(j)
            Wh.append(2 * np.random.randn(n_Zn + 1, self.n_enhancements) - 1)

        #input scaling + clipping
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.clip_min, self.clip_max = percentile_clip_bounds(X_scaled)
        X_scaled = np.clip(X_scaled, self.clip_min, self.clip_max)

        #Quantization Params for X_scaled
        if per_tensor_quant:
            # Per-Tensor: single min/max for the whole array
            X_min_val, X_max_val = np.min(X_scaled), np.max(X_scaled)
            X_min, X_max = np.array([X_min_val]), np.array([X_max_val])
        else:
            # Per-Channel (original): min/max for each column (feature)
            X_min, X_max = np.min(X_scaled, axis = 0), np.max(X_scaled, axis = 0)

        self.S_X, self.Zp_X = compute_quant_params(X_min, X_max, signed=False)

        if QAT:
            X_dequant = fake_quant(X_scaled, self.S_X, self.Zp_X, signed=False)
        else:
            X_dequant = X_scaled

        #add bias feature
        X_biased = np.hstack([X_dequant, 0.1 * np.ones((X.shape[0],1))])

        #define Zn
        Zn = np.zeros((X.shape[0], n_Zn))
        
        #define We
        We_sparse = []
        self.S_We = [None] * self.n_ftr_groups
        self.Zp_We = [None] * self.n_ftr_groups

        #for linear mapping (phi)
        win_dist = np.zeros(self.n_ftr_groups)
        win_mean = np.zeros(self.n_ftr_groups)

        #calculating Zn
        for i in range(self.n_ftr_groups):
            Zi_temp = np.dot(X_biased, We[i])   #X.W
            
            scaler = MinMaxScaler(feature_range=(-1,1))      #scale Zi to range (-1, 1)
            Zi_temp_scaled = scaler.fit_transform(Zi_temp)

            We_sparse.append(ADMM(X_biased, Zi_temp_scaled, self.shrink))   #update Wei and make it sparse, then append it to We (its learning the weights for one iteration, sparse AE)
            
            #Quantization Params for We
            We_i = We_sparse[i]
            We_i_min, We_i_max = np.min(We_i, axis=0), np.max(We_i, axis=0)
            self.S_We[i], self.Zp_We[i] = compute_quant_params(We_i_min, We_i_max, signed=False)

            if QAT:
                We_i_dequant = fake_quant(We_i, self.S_We[i], self.Zp_We[i], signed=False)
            else:
                We_i_dequant = We_i
                        
            Zi_new = np.dot(X_biased, We_i_dequant)   #calculate Zi with new updated sparse weights
            
            win_dist[i], win_mean[i]= Zi_new.max() - Zi_new.min(), Zi_new.mean()
            win_dist[i] = np.where(win_dist[i] < 1e-9, 1e-9, win_dist[i])
            Zi = (Zi_new - win_mean[i]) / win_dist[i]   #linear mapping function (phi)

            Zn[:, self.n_features * i : self.n_features * (i + 1)] = Zi

        #Quantization Params for Zn
        if per_tensor_quant:
            # Per-Tensor: single min/max for the whole Zn array
            Zn_min_val, Zn_max_val = np.min(Zn), np.max(Zn)
            Zn_min, Zn_max = np.array([Zn_min_val]), np.array([Zn_max_val])
        else:
            # Per-Channel (original): min/max for each column (feature node)
            Zn_min, Zn_max = np.min(Zn, axis = 0), np.max(Zn, axis = 0)
            
        self.S_Zn, self.Zp_Zn = compute_quant_params(Zn_min, Zn_max, signed = False)

        if QAT:
            Zn_dequant = fake_quant(Zn, self.S_Zn, self.Zp_Zn, signed=False)
        else:
            Zn_dequant = Zn

        #making enhancement nodes
        Zn_biased = np.hstack([Zn_dequant, 0.1 * np.ones((X.shape[0], 1))])

        Hn = np.zeros((X.shape[0], self.n_enhancements * self.n_enh_groups))
        self.S_Wh = [None] * self.n_enh_groups
        self.Zp_Wh = [None] * self.n_enh_groups

        for j in range(self.n_enh_groups):
            Wh_j = Wh[j]
            Wh_j_min, Wh_j_max = np.min(Wh_j, axis=0), np.max(Wh_j, axis=0)
            self.S_Wh[j], self.Zp_Wh[j] = compute_quant_params(Wh_j_min, Wh_j_max, signed=False)

            if QAT:
                Wh_j_dequant = fake_quant(Wh_j, self.S_Wh[j], self.Zp_Wh[j], signed=False)
            else:
                Wh_j_dequant = Wh_j
            
            Hn[:,j*self.n_enhancements : (j+1)*self.n_enhancements] = self.activation(np.dot(Zn_biased, Wh_j_dequant))

        #Quantization Params for Hn
        if per_tensor_quant:
            # Per-Tensor: single min/max for the whole Hn array
            Hn_min_val, Hn_max_val = np.min(Hn), np.max(Hn)
            Hn_min, Hn_max = np.array([Hn_min_val]), np.array([Hn_max_val])
        else:
            # Per-Channel (original): min/max for each column (enhancement node)
            Hn_min, Hn_max = np.min(Hn, axis = 0), np.max(Hn, axis = 0)
            
        self.S_Hn, self.Zp_Hn = compute_quant_params(Hn_min, Hn_max, signed = False)

        if QAT:
            Hn_dequant = fake_quant(Hn, self.S_Hn, self.Zp_Hn, signed=False)
        else:
            Hn_dequant = Hn

        #making An
        An_float = np.hstack([Zn, Hn])
        #making An_dequant
        An_dequant = np.hstack([Zn_dequant, Hn_dequant])

        #Fake-Quantization parameters
        #QAT params for An_float (used in PTQ test only)
        An_min = np.min(An_float, axis=0)
        An_max = np.max(An_float, axis=0)
        self.S_An, self.Zp_An = compute_quant_params(An_min, An_max, signed = False)

        #calculating Wn = An+ . y
        Wn = np.dot(pseudoinverse(An_dequant, self.lambd), y)      

        #QAT params for Wn
        Wn_min = np.min(Wn, axis = 0)
        Wn_max = np.max(Wn, axis = 0)
        self.S_Wn, self.Zp_Wn = compute_quant_params(Wn_min, Wn_max, signed=False)

        #stop time
        fit_time = time.time() - time_start

        #calculating errors
        y_pred = np.dot(An_float, Wn)
        self.fit_RMSE = RMSE_func(y, y_pred)
        self.fit_MAPE = MAPE_func(y, y_pred)

        self.Wn = Wn
        self.We = We_sparse
        self.Wh = Wh
        self.win_dist = win_dist
        self.win_mean = win_mean
        self.An = An_float
        self.pred_y = y_pred

        self.fit_time = fit_time

        return True

    def predict(self, X, y=None, PTQ = False):
        #start timing
        time_start = time.time()

        #prediction with Post-Training Quantization (u sould use it when QAT was used in the Fit)
        if PTQ:
            X_scaled = self.scaler.transform(X)
            X_scaled = np.clip(X_scaled, self.clip_min, self.clip_max)
            X_dequant = fake_quant(X_scaled, self.S_X, self.Zp_X, signed=False)
            X_biased = np.hstack([X_dequant, 0.1 * np.ones((X.shape[0], 1))])

            Zn = np.zeros((X.shape[0], self.n_features * self.n_ftr_groups))
            for i in range(self.n_ftr_groups):
                We_i_dequant = fake_quant(self.We[i], self.S_We[i], self.Zp_We[i], signed=False)
                Zi_temp = np.dot(X_biased, We_i_dequant)
                Zi = (Zi_temp - self.win_mean[i]) / self.win_dist[i]
                Zn[:, self.n_features * i : self.n_features * (i+1)] = Zi

            Zn_dequant = fake_quant(Zn, self.S_Zn, self.Zp_Zn, signed = False)
            Zn_biased = np.hstack([Zn_dequant, 0.1 * np.ones((Zn.shape[0], 1))])

            Hn = np.zeros((X.shape[0], self.n_enh_groups * self.n_enhancements))
            for j in range(self.n_enh_groups):
                Wh_j_dequant = fake_quant(self.Wh[j], self.S_Wh[j], self.Zp_Wh[j], signed=False)
                Hn[:, j * self.n_enhancements: (j+1) * self.n_enhancements] = self.activation(np.dot(Zn_biased, Wh_j_dequant))

            Hn_dequant = fake_quant(Hn, self.S_Hn, self.Zp_Hn, signed=False)

            An_dequant = np.hstack([Zn_dequant, Hn_dequant])
            
            Wn_dequant = fake_quant(self.Wn, self.S_Wn, self.Zp_Wn, signed=False)

            y_pred = np.dot(An_dequant, Wn_dequant)
        
        #normal prediction whitout any quantization in this stage
        else:
            #input scaling + cliping
            X_scaled = self.scaler.transform(X)
            X_scaled = np.clip(X_scaled, self.clip_min, self.clip_max)
            
            #adding bias to scaled input
            X_biased = np.hstack([X_scaled, 0.1 * np.ones((X.shape[0], 1))])

            #calculating each Zi (feature group) and storing in Zn
            Zn = np.zeros((X.shape[0], self.n_features * self.n_ftr_groups))
            for i in range(self.n_ftr_groups):
                Zi_temp = np.dot(X_biased, self.We[i])
                Zi = (Zi_temp - self.win_mean[i]) / self.win_dist[i]
                Zn[:, self.n_features * i : self.n_features * (i+1)] = Zi

            #adding bias to Zn (for calculating Hn)
            Zn_biased = np.hstack([Zn, 0.1 * np.ones((Zn.shape[0], 1))])

            #calculating each Hi (enhancement group) and storing in Hn
            Hn = np.zeros((X.shape[0], self.n_enh_groups * self.n_enhancements))
            for j in range(self.n_enh_groups):
                Hn[:, j * self.n_enhancements: (j+1) * self.n_enhancements] = self.activation(np.dot(Zn_biased, self.Wh[j]))

            #making An by concatenation
            An = np.hstack([Zn, Hn])

            #calculating prediction
            y_pred = np.dot(An, self.Wn)

        #end of time
        pred_time = time.time() - time_start

        #calculating errors
        if y is not None:
            self.pred_RMSE = RMSE_func(y, y_pred)
            self.pred_MAPE = MAPE_func(y, y_pred)
        
        self.pred_time = pred_time
        self.pred_y = y_pred
        
        return y_pred

    #this function does not have the quantization yet :))
    #increasing the knowledge of the BLS model with new data (by updating the Wn)
    def fit_incremental(self, Xa, y):
        #start time
        time_start = time.time()

        #input scaling + clipping
        Xa_scaled = self.scaler.transform(Xa)
        Xa_scaled = np.clip(Xa_scaled, self.clip_min, self.clip_max)

        #adding bias to input
        Xa_biased = np.hstack([Xa_scaled, 0.1 * np.ones(shape=(Xa.shape[0], 1))])
        
        #calculating each Zi (feature group) and storing in Zn
        n_Zn = self.n_features * self.n_ftr_groups
        Zn_new = np.zeros((Xa.shape[0], n_Zn))
        for i in range(self.n_ftr_groups):
            Zi_temp = np.dot(Xa_biased, self.We[i])
            Zi = (Zi_temp - self.win_mean[i]) / self.win_dist[i]
            Zn_new[:, self.n_features*i:self.n_features*(i+1)] = Zi

        #adding bias to Zn (for calculating Hn)
        Zn_biased = np.hstack([Zn_new, 0.1 * np.ones((Zn_new.shape[0], 1))])

        #calculating each Hi (enhancement group) and storing in Hn
        Hn_new = np.zeros((Xa.shape[0], self.n_enh_groups * self.n_enhancements))
        for j in range(self.n_enh_groups):
            Hn_new[:, j * self.n_enhancements: (j+1) * self.n_enhancements] = self.activation(np.dot(Zn_biased, self.Wh[j]))

        #calculating Ax (that is new data for learning)
        Ax = np.hstack([Zn_new, Hn_new]) 

        #calculating new An (that is new data and old knowledge)
        An_new = np.vstack([self.An, Ax])

        #calculating new knowledge (based on paper)
        An_psi = pseudoinverse(self.An, self.lambd)
        DT = Ax @ An_psi
        CT = Ax - (DT @ self.An)
        if np.allclose(CT, 0, atol=1e-8):
            B = An_psi @ DT.T @ np.linalg.inv(np.eye(DT.shape[0]) + np.dot(DT, DT.T))
        else:
            B = pseudoinverse(CT, self.lambd)

        An_new_psi = np.hstack([An_psi - (B @ DT), B])
        
        #updating Wn (weights of last layer)
        wn = self.Wn.reshape(-1,1)
        y = y.reshape(-1,1)
        Wn_new = wn + B @ (y - Ax @ wn)

        #end of time
        self.fit_inc_time = time.time() - time_start

        if np.any(np.isnan(Wn_new)) or np.any(np.isinf(Wn_new)):
            print('NaN or Inf detected in Wn!')

        self.An = An_new
        self.Wn = Wn_new.reshape(len(Wn_new))
        self.pred_y = np.dot(Ax , Wn_new)
        self.inc_RMSE = RMSE_func(y, self.pred_y)
        self.inc_MAPE = MAPE_func(y, self.pred_y)
        
        return True

    #tunned and annotated by AI (Gemini)
    def fit_transfer(self, X, y, QAT = False, per_tensor_quant = False):
        time_start = time.time()
        
        # --- 1. Mode-Specific Preprocessing & Input Quantization ---
        # NOTE: We recalculate the scaler and clipping bounds for the NEW mode.
        # This overwrites the base model's parameters, so this BLS object now
        # represents the NEW mode (e.g., M1, M2, etc.).
        
        # 1a. Fit NEW Scaler/Standardization
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # 1b. Calculate NEW Clipping Bounds
        self.clip_min, self.clip_max = percentile_clip_bounds(X_scaled)
        X_scaled = np.clip(X_scaled, self.clip_min, self.clip_max)

        # 1c. Calculate NEW Input Quantization Params (S_X, Zp_X)
        if per_tensor_quant:
            # Per-Tensor: single min/max for the whole array
            X_min_val, X_max_val = np.min(X_scaled), np.max(X_scaled)
            X_min, X_max = np.array([X_min_val]), np.array([X_max_val])
        else:
            # Per-Channel: min/max for each column (feature)
            X_min, X_max = np.min(X_scaled, axis = 0), np.max(X_scaled, axis = 0)
        
        self.S_X, self.Zp_X = compute_quant_params(X_min, X_max, signed=False)

        if QAT:
            # Fake quantize the input using the NEW mode's S_X/Zp_X
            X_dequant = fake_quant(X_scaled, self.S_X, self.Zp_X, signed=False)
        else:
            X_dequant = X_scaled

        # Add bias feature
        X_biased = np.hstack([X_dequant, 0.1 * np.ones((X.shape[0],1))])
        n_Zn = self.n_features * self.n_ftr_groups
        Zn = np.zeros((X.shape[0], n_Zn))
        
        # --- 2. Feature Nodes (Zn) Calculation with FROZEN Weights ---
        We_sparse = self.We # Use the FROZEN We weights (set from base model)
        self.win_dist = np.zeros(self.n_ftr_groups) # Recalculate stats
        self.win_mean = np.zeros(self.n_ftr_groups) # Recalculate stats

        for i in range(self.n_ftr_groups):
            # QAT: Use the shared, fixed S_We/Zp_We calculated in the base model fit()
            We_i_dequant = fake_quant(We_sparse[i], self.S_We[i], self.Zp_We[i], signed=False)
                        
            Zi_new = np.dot(X_biased, We_i_dequant)   # Zi uses frozen, quantized We
            
            # Recalculate Z-score stats based on the new data flow
            self.win_dist[i], self.win_mean[i]= Zi_new.max() - Zi_new.min(), Zi_new.mean()
            self.win_dist[i] = np.where(self.win_dist[i] < 1e-9, 1e-9, self.win_dist[i])
            Zi = (Zi_new - self.win_mean[i]) / self.win_dist[i]   # Linear mapping (phi)

            Zn[:, self.n_features * i : self.n_features * (i + 1)] = Zi

        # --- 3. Recalculate Zn Quantization Parameters ---
        if per_tensor_quant:
            # Per-Tensor: single min/max for the whole Zn array
            Zn_min_val, Zn_max_val = np.min(Zn), np.max(Zn)
            Zn_min, Zn_max = np.array([Zn_min_val]), np.array([Zn_max_val])
        else:
            # Per-Channel: min/max for each column
            Zn_min, Zn_max = np.min(Zn, axis = 0), np.max(Zn, axis = 0)
    
        self.S_Zn, self.Zp_Zn = compute_quant_params(Zn_min, Zn_max, signed = False)

        if QAT:
            # Fake quantize the Zn activation using the NEW mode's S_Zn/Zp_Zn
            Zn_dequant = fake_quant(Zn, self.S_Zn, self.Zp_Zn, signed=False)
        else:
            Zn_dequant = Zn

        # --- 4. Enhancement Nodes (Hn) Calculation with FROZEN Weights ---
        Zn_biased = np.hstack([Zn_dequant, 0.1 * np.ones((X.shape[0], 1))])
        Wh = self.Wh # Use the FROZEN Wh weights
        
        Hn = np.zeros((X.shape[0], self.n_enhancements * self.n_enh_groups))
        
        for j in range(self.n_enh_groups):
            # QAT: Use the shared, fixed S_Wh/Zp_Wh calculated in the base model fit()
            Wh_j_dequant = fake_quant(Wh[j], self.S_Wh[j], self.Zp_Wh[j], signed=False)
            
            Hn[:,j*self.n_enhancements : (j+1)*self.n_enhancements] = self.activation(np.dot(Zn_biased, Wh_j_dequant))

        # --- 5. Recalculate Hn Quantization Parameters ---
        if per_tensor_quant:
            # Per-Tensor: single min/max for the whole Hn array
            Hn_min_val, Hn_max_val = np.min(Hn), np.max(Hn)
            Hn_min, Hn_max = np.array([Hn_min_val]), np.array([Hn_max_val])
        else:
            # Per-Channel: min/max for each column
            Hn_min, Hn_max = np.min(Hn, axis = 0), np.max(Hn, axis = 0)
    
        self.S_Hn, self.Zp_Hn = compute_quant_params(Hn_min, Hn_max, signed = False)

        if QAT:
            # Fake quantize the Hn activation using the NEW mode's S_Hn/Zp_Hn
            Hn_dequant = fake_quant(Hn, self.S_Hn, self.Zp_Hn, signed=False)
        else:
            Hn_dequant = Hn

        # --- 6. Output Head (Wn) Retraining & Quantization ---
        An_dequant = np.hstack([Zn_dequant, Hn_dequant])

        # Recalculate Wn = An+ . y
        Wn = np.dot(pseudoinverse(An_dequant, self.lambd), y)      

        # Recalculate QAT params for Wn (Mode-Specific)
        Wn_min = np.min(Wn, axis = 0)
        Wn_max = np.max(Wn, axis = 0)
        self.S_Wn, self.Zp_Wn = compute_quant_params(Wn_min, Wn_max, signed=False)

        # Store new Wn and time
        self.Wn = Wn
        fit_time = time.time() - time_start
        self.fit_time = fit_time
        
        # Calculate errors (optional)
        y_pred = self.predict(X, PTQ=QAT)
        self.fit_RMSE = RMSE_func(y, y_pred)
        self.fit_MAPE = MAPE_func(y, y_pred)
        
        return True
    
    #tunned and annotated by AI (Gemini)
    def export_mode_config(self, monitored_feature_idx, reference_buffer, ks_limit):
        """
        
        Gathers all mode-specific parameters for C code export.
        it is used to make the .h files for each model that containes parameters of that model
        
        """
        config = {
            # --- Inference Parameters (Normalization/Clipping) ---
            "scaler_mean": self.scaler.mean_, 
            "scaler_std": self.scaler.scale_, 
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            
            # --- Quantization & Scaling ---
            "S_X": self.S_X,
            "Zp_X": self.Zp_X,
            "win_mean": self.win_mean,
            "win_dist": self.win_dist,
            "S_Zn": self.S_Zn,
            "Zp_Zn": self.Zp_Zn,
            "S_Hn": self.S_Hn,
            "Zp_Hn": self.Zp_Hn,
            "S_Wn": self.S_Wn,
            "Zp_Wn": self.Zp_Wn,
            
            # --- Output Head Weight ---
            "Wn_quant": np.round(self.Wn / self.S_Wn + self.Zp_Wn).astype(np.uint8),
            
            # --- Drift Detection Parameters ---
            "Monitored_Feature_Idx": monitored_feature_idx,
            "Reference_Buffer": reference_buffer, # This will be the pre-sorted uint8_t array
            "KS_Limit": ks_limit
        }
        return config
        
    #transfer learning without any quantization
    def fit_transfer_old(self, X, y):
        #start time
        time_start = time.time()

        #input scaling + clipping
        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, self.clip_min, self.clip_max)

        #adding bias to input
        X_Biased = np.hstack([X_scaled, 0.1 * np.ones(shape=(X.shape[0],1))])

        #calculating Zn and new parameters for linear mapping function (phi)
        win_dist, win_mean= np.zeros(self.n_ftr_groups), np.zeros(self.n_ftr_groups)
        Zn = np.zeros((X.shape[0], self.n_features * self.n_ftr_groups))
        for i in range(self.n_ftr_groups):
            Zi = np.dot(X_Biased, self.We[i])
            win_dist[i], win_mean[i]= Zi.max() - Zi.min(), Zi.mean()
            Zi_scaled = (Zi - win_mean[i]) / win_dist[i]
            Zn[:, i*self.n_features: (i+1)*self.n_features] = Zi_scaled
        
        #adding bias to Zn
        Zn_biased = np.hstack([Zn, 0.1*np.ones((Zn.shape[0], 1))])

        #calculating Hn
        Hn = np.zeros((X.shape[0], self.n_enh_groups * self.n_enhancements))
        for j in range(self.n_enh_groups):
            Hn[:, j * self.n_enhancements: (j+1) * self.n_enhancements] = self.activation(np.dot(Zn_biased, self.Wh[j]))

        #calculating An
        An = np.hstack([Zn, Hn])

        #calculating new Wn (Wn = An+ . y)
        Wn = np.dot(pseudoinverse(An, self.lambd), y)

        #end of time
        self.fit_trns_time = time.time() - time_start

        if np.any(np.isnan(Wn)) or np.any(np.isinf(Wn)):
            print('NaN or Inf detected in Wn!')

        y_pred = np.dot(An, Wn)

        #calculating errors
        self.trns_RMSE = RMSE_func(y, y_pred)
        self.trns_MAPE = MAPE_func(y, y_pred)

        self.Wn = Wn
        self.An = An
        self.pred_y = y_pred

        self.win_dist = win_dist
        self.win_mean = win_mean

        return True
    
    #plain deift detector
    def ks_drift_detector2(self, old_buffer, new_buffer):
        #sorting old and new buffers
        A = np.sort(old_buffer)
        B = np.sort(new_buffer)
        
        index_A, index_B, count_A, count_B, D = 0, 0, 0, 0, 0
        n = len(new_buffer)
        
        while index_A != n and index_B != n and D != self.limit:
            if A[index_A] > B[index_B]:
                count_B += 1
                index_B += 1
                if abs(count_A - count_B) > D:
                    D += 1

            elif A[index_A] < B[index_B]:
                count_A += 1
                index_A += 1
                if abs(count_A - count_B) > D:
                    D += 1

            else:
                count_A += 1
                count_B += 1
                index_A += 1
                index_B += 1
        
        return D >= self.limit

    #drift detector with sensitivity parameter
    def ks_drift_detector3(self, old_buffer, new_buffer, sensitivity):
        #sorting old and new buffers
        A = np.sort(old_buffer)
        B = np.sort(new_buffer)
        
        index_A, index_B, count_A, count_B, D = 0, 0, 0, 0, 0
        n = len(new_buffer)
        while index_A != n and index_B != n and D != self.limit:
            if A[index_A] > (sensitivity + 1) * B[index_B]:
                count_B += 1
                index_B += 1
                if abs(count_A - count_B) > D:
                    D += 1

            elif A[index_A] * (sensitivity + 1) < B[index_B]:
                count_A += 1
                index_A += 1
                if abs(count_A - count_B) > D:
                    D += 1

            else:
                count_A += 1
                count_B += 1
                index_A += 1
                index_B += 1
            
        return D >= self.limit

    #increasing the number of enhancement groups without retraining
    def inc_enhancement_groups(self, n_new_enh_groups, y):
        #start time
        time_start = time.time()

        #calculating new Hi groups
        Zn = self.An[:, :self.n_features * self.n_ftr_groups]
        Zn_biased = np.hstack([Zn, 0.1*np.ones((Zn.shape[0], 1))])
        Hnp = np.zeros((Zn.shape[0], self.n_enhancements * n_new_enh_groups))
        for i in range(n_new_enh_groups):
            Whp_temp = 2 * np.random.randn(self.n_features * self.n_ftr_groups + 1, self.n_enhancements) - 1
            self.Wh.append(Whp_temp)
            Hnp[:, i * self.n_enhancements : (i+1) * self.n_enhancements] = self.activation(np.dot(Zn_biased, Whp_temp))

        #updating An
        An_new = np.hstack([self.An, Hnp])

        #calculating new Wn
        An_psi = pseudoinverse(self.An, self.lambd)

        D = An_psi @ Hnp
        C = Hnp - (self.An @ D)

        if np.allclose(C, 0, atol=1e-8):
            BT = np.linalg.inv(np.eye(D.shape[1]) + np.dot(D.T, D)) @ D.T @ An_psi
        else:
            BT = pseudoinverse(C, self.lambd)

        An_new_psi = np.vstack([An_psi - (D @ BT), BT])

        Wn_new = np.vstack([self.Wn.reshape(-1,1) - (D @ BT @ y.reshape(-1,1)), BT @ y.reshape(-1,1)])
        
        #end of time
        self.inc_enh_time = time.time() - time_start

        self.An = An_new
        self.Wn = Wn_new
        self.n_enh_groups += n_new_enh_groups
        
        #calculating errors
        self.fit_RMSE = RMSE_func(y, An_new @ Wn_new)
        self.fit_MAPE = MAPE_func(y, An_new @ Wn_new)

        return True
    
    #increasing the number of feature groups and enhancement groups without retraining
    def inc_feature_groups(self, n_new_ftr_groups, n_new_enh_groups, X, y):
        #start time
        time_start = time.time()

        Zn = self.An[:, :self.n_features * self.n_ftr_groups]

        #input scaling + clipping
        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, self.clip_min, self.clip_max)

        #adding bias to input
        X_biased = np.hstack([X_scaled, 0.1 * np.ones((X.shape[0],1))])
        
        #expanding We for new feature groups
        Wep = []
        for i in range(n_new_ftr_groups):
            np.random.seed(i)
            Wep_temp = 2 * np.random.randn(X.shape[1] + 1, self.n_features) - 1
            Wep.append(Wep_temp)
            self.We.append(Wep_temp)
        
        #expanding and calculating new feature groups
        Znp = np.zeros((X.shape[0], self.n_features * n_new_ftr_groups))

        win_dist_p = np.zeros(n_new_ftr_groups)
        win_mean_p = np.zeros(n_new_ftr_groups)

        for i in range(n_new_ftr_groups):
            Zip_temp = X_biased @ Wep[i]
            win_dist_p[i] = Zip_temp.max() - Zip_temp.min()
            win_mean_p[i] = Zip_temp.mean()
            Zip = (Zip_temp - win_mean_p[i]) / win_dist_p[i]
            Znp[:, i * self.n_features : (i+1) * self.n_features] = Zip

        Zn_new = np.hstack([Zn, Znp])

        #expanding and calculating new enhancement groups
        Znp_biased = np.hstack([Znp, 0.1 * np.ones((Znp.shape[0], 1))])

        Hnp = np.zeros((X.shape[0], n_new_enh_groups * self.n_enhancements))
        Whp = []
        for j in range(n_new_enh_groups):
            Whp_temp = 2 * np.random.randn(Znp.shape[1]+1, self.n_enhancements) - 1
            Whp.append(Whp_temp)
            self.Wh.append(Whp_temp)
            Hnp[:, j * self.n_enhancements : (j+1) * self.n_enhancements] = self.activation(Znp_biased @ Whp_temp)

        #calculating new Wn
        Anp = np.hstack([Znp, Hnp])
        An_new = np.hstack([self.An, Anp])

        An_psi = pseudoinverse(self.An, self.lambd)
        D = An_psi @ Anp
        C = Anp - (self.An @ D)

        if np.allclose(C, 0, atol=1e-8):
            BT = np.linalg.inv(np.eye(D.shape[1]) + np.dot(D.T, D)) @ D.T @ An_psi
        else:
            BT = pseudoinverse(C, self.lambd)

        An_new_psi = np.vstack([An_psi - (D @ BT), BT])
        Wn_new = np.vstack([self.Wn.reshape(-1,1) - (D @ BT @ y.reshape(-1,1)), BT @ y.reshape(-1,1)])

        self.inc_ftr_time = time.time() - time_start

        self.An = An_new
        self.Wn = Wn_new
        # self.We.append(Wep)

        # Wh_new = np.zeros((self.Wh.shape[0]+Whp.shape[0]-1, self.Wh.shape[1]+Whp.shape[1]))
        # Wh_new[:self.Wh.shape[0],:self.Wh.shape[1]] = self.Wh
        # Wh_new[self.Wh.shape[0]:,self.Wh.shape[1]:] = Whp[:Whp.shape[0]-1,:]
        # self.Wh = Wh_new
        # self.Wh = np.vstack([self.Wh , Whp])
        y_pred = An_new @ Wn_new

        #calculating errors
        self.fit_RMSE = RMSE_func(y, y_pred)
        self.fit_MAPE = MAPE_func(y, y_pred)
        self.win_dist = np.hstack([self.win_dist, win_dist_p])
        self.win_mean = np.hstack([self.win_mean, win_mean_p])
        self.n_ftr_groups += n_new_ftr_groups
        self.n_enh_groups += n_new_enh_groups

        return True

    