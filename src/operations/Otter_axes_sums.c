#include "../../header/operations/Otter_axes_sums.h"

OtterTensor* OT_column_sum(OtterTensor* t) {
    if (t->rank != 2) {
        fprintf(stderr, "Column sum is only defined for 2D tensors.\n");
        exit(EXIT_FAILURE);
    }
    
    OtterTensor* result = OT_zeros((int[2]){t->dims[1],1}, 2);
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int j = 0; j < t->dims[1]; j++) {
        for (int i = 0; i < t->dims[0]; i++) {
            result->data[j] += t->data[j + i * t->dims[1]];
        }
    }
    #else
    for (int j = 0; j < t->dims[1]; j++) {
        for (int i = 0; i < t->dims[0]; i++) {
            result->data[j] += t->data[j + i * t->dims[1]];
        }
    }
    #endif
    return result;
}

OtterTensor* OT_line_sum(OtterTensor* t) {
    if (t->rank != 2) {
        fprintf(stderr, "line sum is only defined for 2D tensors.\n");
        exit(EXIT_FAILURE);
    }
    
    OtterTensor* result = OT_zeros((int[2]){t->dims[0],1}, 2);
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int j = 0; j < t->dims[0]; j++) {
        for (int i = 0; i < t->dims[1]; i++) {
            result->data[j] += t->data[j*t->dims[1]+i];
        }
    }
    #else
    for (int j = 0; j < t->dims[0]; j++) {
        for (int i = 0; i < t->dims[1]; i++) {
            result->data[j] += t->data[j*t->dims[1]+i];
        }
    }
    #endif
    return result;
}

float OT_sum(OtterTensor* t) {
    float total = 0.0f;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < t->size; i++) {
        total += t->data[i];
    }
    #else
    for (int i = 0; i < t->size; i++) {
        total += t->data[i];
    }
    #endif
    return total;
}