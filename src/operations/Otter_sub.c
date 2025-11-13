#include "../../header/operations/Otter_sub.h"

OtterTensor* OT_tensors_substract(OtterTensor* a, OtterTensor* b) {
    OtterTensor* result=OT_zeros(a->dims, a->rank);
    if (a->rank != b->rank || a->rank != result->rank) {
        fprintf(stderr, "Tensors must have the same rank for subtraction.\n");
        exit(EXIT_FAILURE);
    }
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    #else
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    #endif
    return result;
}


void OT_ref_tensors_substract(OtterTensor* a, OtterTensor* b) {

    if (a->size != b->size) {
        fprintf(stderr, "[Error in ] Tensor sizes do not match for addition: %d vs %d\n",  a->size, b->size);
        printf("Tensor A:\n");
        print_tensor(a,2);
        printf("Tensor B:\n");
        print_tensor(b,2);
        exit(EXIT_FAILURE);
    }

    if (a->rank != b->rank ) {
        fprintf(stderr, "Tensors must have the same rank for subtraction.\n");
        exit(EXIT_FAILURE);
    }
    
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < a->size; i++) {
        a->data[i] = a->data[i] - b->data[i];
    }

    #else
    for (int i = 0; i < a->size; i++) {
        a->data[i] = a->data[i] - b->data[i];
    }
    #endif

    return;
}
