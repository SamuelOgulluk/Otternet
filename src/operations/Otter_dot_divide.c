#include "../../header/operations/Otter_dot_divide.h"

void OT_ref_dot_divide(OtterTensor* dividend, OtterTensor* divisor) {
    if (dividend->size != divisor->size) {
        fprintf(stderr, "Tensors must have the same size for dot division.\n");
        exit(EXIT_FAILURE);
    }
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < dividend->size; i++) {
        if (divisor->data[i] == 0.0f) {
            fprintf(stderr, "Error: Division by zero in OT_ref_dot_divide at index %d\n", i);
            dividend->data[i] = 0.0f; // or NAN
        } else {
            dividend->data[i] /= divisor->data[i];
        }
    }
    #else
    for (int i = 0; i < dividend->size; i++) {
        if (divisor->data[i] == 0.0f) {
            fprintf(stderr, "Error: Division by zero in OT_ref_dot_divide at index %d\n", i);
            dividend->data[i] = 0.0f; // or NAN
        } else {
            dividend->data[i] /= divisor->data[i];
        }
    }
    #endif
    return;
}


OtterTensor* OT_dot_divide(OtterTensor* main, OtterTensor* divisor) {
    OtterTensor* result = OT_zeros(main->dims, main->rank);
    OT_ref_dot_divide(result, divisor);
    return result;
}