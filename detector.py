import numpy as np

def detector_maker(buffer_size, sensitivity=0):
    limit = int(np.ceil(np.sqrt(np.log(0.025) * -1 * buffer_size)))
    sensitivity_param = sensitivity + 1
    
    def ks_drift_detector(old_buffer, new_buffer):
        #sorting old and new buffers
        A = np.sort(old_buffer)
        B = np.sort(new_buffer)
        
        index_A, index_B, count_A, count_B, D = 0, 0, 0, 0, 0
        n = len(new_buffer)
        while index_A != n and index_B != n and D != limit:
            if A[index_A] > sensitivity_param * B[index_B]:
                count_B += 1
                index_B += 1
                if abs(count_A - count_B) > D:
                    D += 1

            elif A[index_A] * sensitivity_param < B[index_B]:
                count_A += 1
                index_A += 1
                if abs(count_A - count_B) > D:
                    D += 1

            else:
                count_A += 1
                count_B += 1
                index_A += 1
                index_B += 1
            
        return D >= limit
    return ks_drift_detector