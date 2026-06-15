// File: drift_detector.h
#include <stdbool.h>
#include <stdint.h>

#ifndef DRIFT_DETECTOR_H
#define DRIFT_DETECTOR_H

// Type definitions for clarity
typedef uint8_t DataValue;

// Helper function for qsort
int compare_uint8(const void *a, const void *b);

/**
 * @brief Runs the integer-only KS test between two buffers.
 * * @param buf_1_sorted   Pointer to the "Baseline" data.
 * In Continuous Mode: This is 'Old_Buffer' (RAM, already sorted).
 * In Search Mode: This is 'Reference_Buffer' (Flash, pre-sorted).
 * @param size_1         Size of buf_1.
 * * @param buf_2_unsorted Pointer to the "Incoming" data (New_Buffer).
 * THIS WILL BE SORTED IN-PLACE by this function.
 * @param size_2         Size of buf_2.
 * * @param limit          The integer D-threshold (KS_LIMIT).
 * * @return true if Drift Detected, false otherwise.
 */
bool ks_drift_detector(
    const DataValue *buf_1_sorted, int size_1,
    DataValue *buf_2_unsorted, int size_2,
    int limit);

#endif // DRIFT_DETECTOR_H