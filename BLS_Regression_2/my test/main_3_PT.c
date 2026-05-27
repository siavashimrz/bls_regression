// File: main.c - Main Application Loop and BLS/Drift Logic
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

// --- Project Headers ---
// Shared weights and constants (We, Wh, S, Zp)
#include "bls_shared_weights_3.h"
// Mode-specific configurations (Wn, Reference Buffers, KS_LIMIT, MONITORED_FEATURE_IDX)
#include "bls_mode_configs_3.h"
// Your finalized drift detection logic
#include "drift_detector_3.h"
// #include "test_batch_data_3.h"
// #include "test_scenario_1_drift.h"
// #include "test_mode0_100.h"
// #include "test_mode1_100.h"
// #include "test_mode2_100.h"
// #include "test_mode3_100.h"
// #include "test_mode4_100.h"
// #include "test_mode5_100.h"
#include "test_all_modes_100.h"

// --- Constants (Derived from bls_mode_configs.h and BLS structure) ---
// Note: These constants should match the generation script's output
#define N_MODES 6
#define INPUT_DIM 16
#define N_FEATURES 10
#define N_FTR_GROUPS 8
#define N_ENHANCEMENTS 6
#define N_ENH_GROUPS 3

#define N_ZN_FEATURES (N_FEATURES * N_FTR_GROUPS)
#define N_HN_FEATURES (N_ENHANCEMENTS * N_ENH_GROUPS)

#define N_WE_PER_GROUP ((INPUT_DIM + 1) * N_FEATURES)         // 17 * 10 = 170
#define N_WH_PER_GROUP ((N_ZN_FEATURES + 1) * N_ENHANCEMENTS) // 81* 6 = 486

// Window size for KS test (defined in bls_mode_configs.h)
#define BUFFER_SIZE 30
#define KS_LIMIT 11

// --- Global State ---
static int8_t current_mode_idx;
// Pointer to the currently active ModeConfig in Flash (or RAM, but conceptually frozen)
static const ModeConfig_t *active_config;

// Buffers for the monitored feature (uint8_t)
static DataValue Monitored_Feature_Buffer_A[BUFFER_SIZE]; // RAM Buffer A (The working buffers)
static DataValue Monitored_Feature_Buffer_B[BUFFER_SIZE]; // RAM Buffer B

// Pointers for the sliding window logic (these pointers will swap)
static DataValue *Old_Buffer_Ptr = Monitored_Feature_Buffer_A;
static DataValue *New_Buffer_Ptr = Monitored_Feature_Buffer_B;
static int new_buffer_count = 0; // Tracks how many samples are in New_Buffer_Ptr

// --- Utility Functions ---

/**
 * @brief Attempts to find the best mode configuration after drift is detected.
 * @param new_sorted_buffer The current, sorted data buffer.
 * @return The index of the best matching mode (0 to N_MODES-1) or the current mode.
 */
void search_best_mode(DataValue *new_sorted_buffer)
{
    printf("Starting Mode Search...\n");

    for (int i = 0; i < N_MODES; i++)
    {
        // Compare the new buffer against the frozen reference buffer of mode 'i'
        const ModeConfig_t *mode_i = &Modes[i];

        bool drift_detected = ks_drift_detector(
            mode_i->Reference_Buffer, BUFFER_SIZE,
            new_sorted_buffer, BUFFER_SIZE,
            KS_LIMIT);

        printf("Mode %d drift detected: %s\n", i, drift_detected ? "True" : "False");

        if (!drift_detected)
        {
            current_mode_idx = i;
            break;
        }
    }
    printf("Best match found: Mode %d.\n", current_mode_idx);
}

/**
 * @brief Initializes the system state.
 */
void initialize_system()
{
    // 1. Set the initial active mode configuration (Mode 0)
    current_mode_idx = 0;
    active_config = &Modes[current_mode_idx];

    // 2. Initialize the "Old" buffer with the Reference Data for Mode 0
    // This provides a clean starting point for the first drift check.
    memcpy(Old_Buffer_Ptr, active_config->Reference_Buffer, BUFFER_SIZE * sizeof(DataValue));

    printf("System Initialized to Mode %d. Buffer Size: %d. KS Limit: %d.\n",
           current_mode_idx, BUFFER_SIZE, KS_LIMIT);
}

/**
 * @brief Main function to process incoming data and manage mode switching.
 * @param X_sample The single input data vector (INPUT_DIM elements).
 * @return The prediction for the sample.
 */
void buffer_and_detection(const float *X_sample)
{
    // 2. Quantize the monitored feature for drift detection
    float monitored_feature = X_sample[MONITORED_FEATURE_IDX];

    // Quantize the monitored feature using the global KS scales (S_KS_Global, Zp_KS_Global)
    // DataValue feature_quant = quantize_value(monitored_feature, S_KS_Global, Zp_KS_Global);
    DataValue feature_quant;

    // 3. Quantization (Formula: q = round(x / S + Zp))
    // We use the input's own quantization params (S_X, Zp_X)
    float quant_val = roundf(monitored_feature / S_KS_Global + Zp_KS_Global);

    // 4. Clip to the 8-bit range [0, 255]
    // We use 'uint8_t' so 0-255 is the range.
    if (quant_val > 255.0f)
    {
        feature_quant = 255;
    }
    else if (quant_val < 0.0f)
    {
        feature_quant = 0;
    }
    else
    {
        feature_quant = (uint8_t)quant_val;
    }

    // 3. Store the quantized feature in the New Buffer
    New_Buffer_Ptr[new_buffer_count] = feature_quant;
    new_buffer_count++;

    // 4. Check for Drift when the buffer is full
    if (new_buffer_count == BUFFER_SIZE)
    {
        // The ks_drift_detector will sort New_Buffer_Ptr in-place.
        // Old_Buffer_Ptr is the previous cycle's sorted data.
        bool drift_detected = ks_drift_detector(
            Old_Buffer_Ptr, BUFFER_SIZE,
            New_Buffer_Ptr, BUFFER_SIZE,
            KS_LIMIT);

        if (drift_detected)
        {
            printf("\n--- DRIFT DETECTED from Mode %d ---\n", current_mode_idx);

            // New_Buffer_Ptr is now sorted, use it for mode search
            search_best_mode(New_Buffer_Ptr);
            // CHANGE MODE
            active_config = &Modes[current_mode_idx];
            printf("!!! MODE SWITCHED to Mode %d. !!!\n", current_mode_idx);

            // When mode switches, the "Old" buffer must be re-initialized
            // to the new mode's Reference Buffer for the next continuous check.
            // This ensures the Old Buffer is representative of the NEW model's expected input.
            // memcpy(Old_Buffer_Ptr, active_config->Reference_Buffer, BUFFER_SIZE * sizeof(DataValue));
        }
        // No Drift Detected: Efficient Sliding Window Logic
        // The sorted New buffer becomes the Old buffer for the next cycle.
        // We swap the pointers to avoid a data copy.
        DataValue *temp = Old_Buffer_Ptr;
        Old_Buffer_Ptr = New_Buffer_Ptr;
        New_Buffer_Ptr = temp; // New_Buffer_Ptr now points to the location that Old_Buffer_Ptr *just* vacated

        // Reset the counter for the New Buffer (which is now empty)
        new_buffer_count = 0;
        printf("\n----------------- Buffer got empty -------------------\n");
    }
}

void quantize_input(const float *in_float, uint8_t *out_uint8)
{
    for (int i = 0; i < INPUT_DIM; i++)
    {
        float scaled;

        // 1. Standardization (Z-score scaling)
        scaled = (in_float[i] - active_config->scaler_mean[i]) / active_config->scaler_std[i];

        // 2. Min/Max Clipping (based on training range)
        scaled = fmaxf(active_config->clip_min[i], fminf(active_config->clip_max[i], scaled));

        // 3. Quantization (Formula: q = round(x / S + Zp))
        // We use the input's own quantization params (S_X, Zp_X)
        float quant_val = roundf(scaled / active_config->S_X[0] + active_config->Zp_X[0]);

        // 4. Clip to the 8-bit range [0, 255]
        // We use 'uint8_t' so 0-255 is the range.
        if (quant_val > 255.0f)
        {
            out_uint8[i] = 255;
        }
        else if (quant_val < 0.0f)
        {
            out_uint8[i] = 0;
        }
        else
        {
            out_uint8[i] = (uint8_t)quant_val;
        }
    }
}

float bls_inference(const float *input_data_float)
{

    // ==================================================================================
    // STEP 1: INPUT PREPROCESSING & QUANTIZATION
    // ==================================================================================

    uint8_t X_q_features[INPUT_DIM];
    quantize_input(input_data_float, X_q_features);

    uint8_t X_q_biased[INPUT_DIM + 1];
    for (int i = 0; i < INPUT_DIM; i++)
    {
        X_q_biased[i] = X_q_features[i];
    }

    // FIX 1: Quantize bias using Zp_X[0]
    // 0.1f is the bias value used in training
    float bias_q_float = 0.1f / active_config->S_X[0] + (float)active_config->Zp_X[0];
    if (bias_q_float > 255.0f)
        bias_q_float = 255.0f;
    if (bias_q_float < 0.0f)
        bias_q_float = 0.0f;
    X_q_biased[INPUT_DIM] = (uint8_t)bias_q_float;

    // ==================================================================================
    // STEP 2: FEATURE NODES (Zn) CALCULATION
    // ==================================================================================

    float Zn_float[N_ZN_FEATURES];
    uint8_t Zn_q[N_ZN_FEATURES];

    for (int g = 0; g < N_FTR_GROUPS; g++)
    {
        const uint8_t *We_g_ptr;
        switch (g)
        {
        case 0:
            We_g_ptr = &We_quant[0];
            break;
        case 1:
            We_g_ptr = &We_quant[170];
            break;
        case 2:
            We_g_ptr = &We_quant[340];
            break;
        case 3:
            We_g_ptr = &We_quant[3 * 170];
            break;
        case 4:
            We_g_ptr = &We_quant[4 * 170];
            break;
        case 5:
            We_g_ptr = &We_quant[5 * 170];
            break;
        case 6:
            We_g_ptr = &We_quant[6 * 170];
            break;
        case 7:
            We_g_ptr = &We_quant[7 * 170];
            break;
        default:
            We_g_ptr = &We_quant[0];
            break;
        }

        int32_t Zi_acc[N_FEATURES];

        for (int n = 0; n < N_FEATURES; n++)
        {
            int32_t acc = 0;

            // Calculate global index for Zp/Scale access
            int global_idx = g * N_FEATURES + n;
            int32_t zp_w_val = (int32_t)Zp_We[global_idx]; // FIX 2: Use global_idx

            for (int i = 0; i < INPUT_DIM; i++)
            {
                int32_t input_val = (int32_t)X_q_biased[i] - (int32_t)active_config->Zp_X[0];
                // We_quant is (17, 10). Access row 'i', col 'n'.
                int32_t weight_val = (int32_t)We_g_ptr[i * N_FEATURES + n] - zp_w_val;
                acc += input_val * weight_val;
            }

            // Bias term (row 16)
            // FIX 3: Subtract Zp_X[0] because that's what we used to quantize it
            int32_t bias_val = (int32_t)X_q_biased[INPUT_DIM] - (int32_t)active_config->Zp_X[0];
            int32_t weight_val_bias = (int32_t)We_g_ptr[INPUT_DIM * N_FEATURES + n] - zp_w_val;
            acc += bias_val * weight_val_bias;

            Zi_acc[n] = acc;

            // De-quantize & Scale
            // FIX 4: Use global_idx for S_We
            float zi_float = (float)Zi_acc[n] * active_config->S_X[0] * S_We[global_idx];

            float zn_val = (zi_float - active_config->win_mean[g]) / active_config->win_dist[g];
            Zn_float[global_idx] = zn_val;

            float quant_temp = roundf(zn_val / active_config->S_Zn[0] + active_config->Zp_Zn[0]);
            if (quant_temp > 255.0f)
                quant_temp = 255.0f;
            if (quant_temp < 0.0f)
                quant_temp = 0.0f;
            Zn_q[global_idx] = (uint8_t)quant_temp;
        }
    }

    // ==================================================================================
    // STEP 3: ENHANCEMENT NODES (Hn) CALCULATION
    // ==================================================================================

    uint8_t Zn_q_biased[N_ZN_FEATURES + 1];
    for (int i = 0; i < N_ZN_FEATURES; i++)
        Zn_q_biased[i] = Zn_q[i];

    // FIX 5: Quantize bias using Zp_Zn[0]
    float bias_zn_q_float = 0.1f / active_config->S_Zn[0] + (float)active_config->Zp_Zn[0];
    if (bias_zn_q_float > 255.0f)
        bias_zn_q_float = 255.0f;
    if (bias_zn_q_float < 0.0f)
        bias_zn_q_float = 0.0f;
    Zn_q_biased[N_ZN_FEATURES] = (uint8_t)bias_zn_q_float;

    float Hn_float[N_HN_FEATURES];
    uint8_t Hn_q[N_HN_FEATURES];

    for (int g = 0; g < N_ENH_GROUPS; g++)
    {
        const uint8_t *Wh_g_ptr;
        switch (g)
        {
        case 0:
            Wh_g_ptr = &Wh_quant[0];
            break;
        case 1:
            Wh_g_ptr = &Wh_quant[1 * 486];
            break;
        case 2:
            Wh_g_ptr = &Wh_quant[2 * 486];
            break;
        default:
            Wh_g_ptr = &Wh_quant[0];
            break;
        }

        for (int n = 0; n < N_ENHANCEMENTS; n++)
        {
            int32_t acc = 0;
            int global_idx = g * N_ENHANCEMENTS + n;
            int32_t zp_wh_val = (int32_t)Zp_Wh[global_idx]; // FIX 6: Use global_idx

            for (int i = 0; i < N_ZN_FEATURES; i++)
            {
                int32_t input_val = (int32_t)Zn_q_biased[i] - (int32_t)active_config->Zp_Zn[0];
                // Wh shape: (81, 6).
                int32_t weight_val = (int32_t)Wh_g_ptr[i * N_ENHANCEMENTS + n] - zp_wh_val;
                acc += input_val * weight_val;
            }
            // Bias term (row 80)
            // FIX 7: Subtract Zp_Zn[0]
            int32_t bias_val = (int32_t)Zn_q_biased[N_ZN_FEATURES] - (int32_t)active_config->Zp_Zn[0];
            int32_t weight_val_bias = (int32_t)Wh_g_ptr[N_ZN_FEATURES * N_ENHANCEMENTS + n] - zp_wh_val;
            acc += bias_val * weight_val_bias;

            // De-quantize
            float hn_temp = (float)acc * active_config->S_Zn[0] * S_Wh[global_idx]; // FIX 8: Use global_idx for Scale

            if (hn_temp < 0.0f)
                hn_temp = 0.0f;
            if (hn_temp > 6.0f)
                hn_temp = 6.0f;

            Hn_float[global_idx] = hn_temp;

            float quant_temp = roundf(hn_temp / active_config->S_Hn[0] + active_config->Zp_Hn[0]);
            if (quant_temp > 255.0f)
                quant_temp = 255.0f;
            if (quant_temp < 0.0f)
                quant_temp = 0.0f;
            Hn_q[global_idx] = (uint8_t)quant_temp;
        }
    }

    // ==================================================================================
    // STEP 4: FINAL PREDICTION (An * Wn)
    // ==================================================================================

    float y_final = 0.0f;

    for (int i = 0; i < N_ZN_FEATURES; i++)
    {
        int32_t input_val = (int32_t)Zn_q[i] - (int32_t)active_config->Zp_Zn[0];
        int32_t weight_val = (int32_t)active_config->Wn_quant[i] - (int32_t)active_config->Zp_Wn[0];
        y_final += (float)(input_val * weight_val) * active_config->S_Zn[0] * active_config->S_Wn[0];
    }

    for (int i = 0; i < N_HN_FEATURES; i++)
    {
        int32_t input_val = (int32_t)Hn_q[i] - (int32_t)active_config->Zp_Hn[0];
        int32_t weight_val = (int32_t)active_config->Wn_quant[N_ZN_FEATURES + i] - (int32_t)active_config->Zp_Wn[0];
        y_final += (float)(input_val * weight_val) * active_config->S_Hn[0] * active_config->S_Wn[0];
    }

    return y_final;
}

void final_table(float *final_rmse, float *final_mape)
{
    printf("\n\n====================================================================\n");
    printf("                       MODE DATA PERFORMANCE SUMMARY\n");
    printf("====================================================================\n");
    printf("|  Data Set  |  Samples |   RMSE   |  MAPE (%%) |\n");
    printf("|------------|----------|----------|------------|\n");

    for (int mode = 0; mode < N_MODES; mode++)
    {
        // Data Set refers to the Mode's data batch (Mode 0 Data, Mode 1 Data, etc.)
        printf("|  Mode %d   |   %4d   | %8.4f | %9.4f  |\n",
               mode,
               N_TEST_SAMPLES,
               final_rmse[mode],
               final_mape[mode]);
    }
    printf("====================================================================\n");
}

// Example main loop (replace with your embedded loop/data acquisition)
int main()
{
    initialize_system();

    // 🚨 ADDITION 1: File Setup and Header Write
    FILE *log_file = fopen("bls_drift_log.csv", "w");
    if (log_file == NULL)
    {
        printf("Error: Could not open log file for writing.\n");
        // We will proceed without logging if file open fails.
    }

    // CSV Header: Sample ID, Data Set Mode (Batch ID), Active Model, Real Target, Predicted Value, Absolute Error
    if (log_file != NULL)
    {
        fprintf(log_file, "Sample_Global,Data_Set_Mode,Active_Model_Index,Real_Target,Predicted_Value,Error_Abs\n");
    }

    printf("--- Starting Prediction Loop ---\n");

    // 🚨 ADDITION 2: Global Sample Counter
    int global_sample_count = 0;

    printf("--- Starting Prediction Loop ---\n");
    float final_rmse[N_MODES];
    float final_mape[N_MODES];
    float prediction;
    float rmse = 0;
    float mape = 0;
    float mean_error = 0;
    float error = 0;
    const float (*X[6])[16] = {test_mode0_input_batch, test_mode1_input_batch, test_mode2_input_batch, test_mode3_input_batch, test_mode4_input_batch, test_mode5_input_batch};
    const float *y[6] = {test_mode0_target_batch, test_mode1_target_batch, test_mode2_target_batch, test_mode3_target_batch, test_mode4_target_batch, test_mode5_target_batch};
    for (int mode = 0; mode < N_MODES; mode++)
    {

        rmse = 0;
        mape = 0;
        mean_error = 0;
        for (int i = 0; i < N_TEST_SAMPLES; i++)
        {
            prediction = bls_inference(X[mode][i]) + active_config->Bias[0];
            error = y[mode][i] - prediction;
            // printf("%d-%d>>\t%f -- %f\terror: %f\n", mode, i, y[mode][i], prediction, error);

            buffer_and_detection(X[mode][i]);

            // 🚨 ADDITION 3: Log Data Line
            if (log_file != NULL)
            {
                fprintf(log_file, "%d,%d,%d,%f,%f,%f\n",
                        global_sample_count, // 1. Global Sample ID (Time)
                        mode,                // 2. Data Set Mode (Batch ID)
                        current_mode_idx,    // 3. The currently Active Model (dynamically changes)
                        y[mode][i],          // 4. Real Target
                        prediction,          // 5. Predicted Value
                        fabsf(error));       // 6. Absolute Error
            }

            // Increment Global Counter
            global_sample_count++;

            rmse = rmse + powf((error), 2.0);
            mape = mape + fabsf((error) / y[mode][i]);
            mean_error += error;
        }
        rmse = sqrt(rmse / N_TEST_SAMPLES);
        mape = (mape / N_TEST_SAMPLES) * 100;
        mean_error /= N_TEST_SAMPLES;
        printf("--- Mode %d Finished ---\n", mode);
        printf("--- RMSE: %f MAPE: %f %% ---\n", rmse, mape);
        // printf("--- RMSE: %f MAPE: %f %% MeanError: %f ---\n", rmse, mape, mean_error);
        final_rmse[mode] = rmse;
        final_mape[mode] = mape;
    }
    final_table(final_rmse, final_mape);

    // 🚨 ADDITION 4: Close the file
    if (log_file != NULL)
    {
        fclose(log_file);
        printf("\nLog file 'bls_drift_log.csv' created successfully.\n");
    }

    printf("\nPress ENTER to exit...\n");
    getchar();
    return 0;
}

// // Example main loop (replace with your embedded loop/data acquisition)
// int main()
// {
//     initialize_system();

//     printf("--- Starting Prediction Loop ---\n");

//     float prediction;
//     float rmse = 0;
//     float mape = 0;
//     float mean_error = 0;

//     for (int i = 0; i < N_TEST_SAMPLES; i++)
//     {
//         // if (i == 70)
//         //     printf("\n>>>> mode 1 started\n\n");
//         prediction = bls_inference(test_input_batch[i]) + active_config->Bias[0];
//         // printf("--> sample %d error:\t%f\n", i, prediction - test_target_batch[i]);
//         // printf("%d>>\t%f -- %f\terror: %f\n", i, test_target_batch[i], prediction, test_target_batch[i] - prediction);

//         buffer_and_detection(test_input_batch[i]);

//         rmse = rmse + powf((test_target_batch[i] - prediction), 2.0);
//         mape = mape + fabsf((test_target_batch[i] - prediction) / test_target_batch[i]);
//         mean_error += test_target_batch[i] - prediction;
//     }
//     rmse = sqrt(rmse / N_TEST_SAMPLES);
//     mape = (mape / N_TEST_SAMPLES) * 100;
//     mean_error /= N_TEST_SAMPLES;
//     printf("--- Loop Finished ---\n");
//     printf("--- RMSE: %f MAPE: %f %% MeanError: %f ---\n", rmse, mape, mean_error);

//     return 0;
// }
