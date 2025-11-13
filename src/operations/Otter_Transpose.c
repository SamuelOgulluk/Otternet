#include "../../header/operations/Otter_transpose.h"

OtterTensor* OT_Transpose(OtterTensor* t) {
    if (t->rank != 2) {
        fprintf(stderr, "Transpose is only defined for 2D tensors.\n");
        exit(EXIT_FAILURE);
    }
    OtterTensor* transposed = OT_zeros((int[2]){t->dims[1], t->dims[0]}, 2);
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < t->dims[0]; i++) {
        for (int j = 0; j < t->dims[1]; j++) {
            transposed->data[j * t->dims[0] + i] = t->data[i * t->dims[1] + j];
        }
    }
    #else
    for (int i = 0; i < t->dims[0]; i++) {
        for (int j = 0; j < t->dims[1]; j++) {
            transposed->data[j * t->dims[0] + i] = t->data[i * t->dims[1] + j];
        }
    }
    #endif
    return transposed;
}