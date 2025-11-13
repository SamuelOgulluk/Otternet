#include "../../header/operations/Otter_dot_prod.h"

OtterTensor* OT_dot(OtterTensor* a, OtterTensor* b) {
    OtterTensor* result = OT_zeros(a->dims, a->rank);
    if (a->rank != b->rank || a->rank != result->rank) {
        fprintf(stderr, "Tensors must have the same rank for multiplication.\n");
        exit(EXIT_FAILURE);
    }
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    #else
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    #endif
    return result;
}