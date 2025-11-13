#include "../../header/operations/Otter_sum.h"
OtterTensor* OT_tensors_sum(OtterTensor* a, OtterTensor* b) {
    OtterTensor* result=OT_zeros(a->dims, a->rank);

    if (a->rank != b->rank || a->rank != result->rank) {
        fprintf(stderr, "Tensors must have the same rank for addition.\n found rank %i and %i \n",a->rank,b->rank);
        exit(EXIT_FAILURE);
    }
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    #else
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    #endif
    return result;
}



void OT_ref_tensors_sum(OtterTensor* a, OtterTensor* b, const char* caller_name) {

    if (a->size != b->size) {
        fprintf(stderr, "[Error in %s] Tensor sizes do not match for addition: %d vs %d\n", caller_name, a->size, b->size);
        printf("Tensor A:\n");
        print_tensor(a,2);
        printf("Tensor B:\n");
        print_tensor(b,2);
        exit(EXIT_FAILURE);
    }

    if (a->rank != b->rank ) {
        fprintf(stderr, "[Error in %s] Tensors must have the same rank for addition. Found rank %i and %i\n", caller_name, a->rank, b->rank);
        exit(EXIT_FAILURE);
    }
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < a->size; i++) {
        a->data[i] = a->data[i] + b->data[i];
    }

    #else
    for (int i = 0; i < a->size; i++) {
        a->data[i] = a->data[i] + b->data[i];
    }
    #endif


    return;
}