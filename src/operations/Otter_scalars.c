#include "../../header/operations/Otter_scalars.h"

void OT_ref_scalar_multiply(OtterTensor* main, float lambda) {
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < main->size; i++) {
        main->data[i] *= lambda;
    }
    #else
    for (int i = 0; i < main->size; i++) {
        main->data[i] *= lambda;
    }
    #endif
    return;
}

OtterTensor* OT_scalar_multiply(OtterTensor* main, float lambda) {
    OtterTensor* result=OT_zeros(main->dims,main->rank);
    #ifdef  _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < main->size; i++) {
        result->data[i] = lambda * main->data[i];
    }
    #else
    for (int i = 0; i < main->size; i++) {
        result->data[i] = lambda * main->data[i];
    }
    #endif
    return result;
}



OtterTensor* OT_scalar_add(OtterTensor* main, float lambda) {
    OtterTensor* result=OT_zeros(main->dims, main->rank);
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < main->size; i++) {
        result->data[i] = main->data[i] + lambda;
    }
    #else
    for (int i = 0; i < main->size; i++) {
        result->data[i] = main->data[i] + lambda;
    }
    #endif
    return result;
}

void OT_ref_scalar_sum(OtterTensor* main, float lambda) {
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < main->size; i++) {
        main->data[i] += lambda;
    }
    #else
    for (int i = 0; i < main->size; i++) {
        main->data[i] += lambda;
    }
    #endif
    return;
}

OtterTensor* OT_scalar_subtract(OtterTensor* main, float lambda) {
    OtterTensor* result=OT_zeros(main->dims,main->rank);
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < main->size; i++) {
        result->data[i] = main->data[i] - lambda;
    }
    #else
    for (int i = 0; i < main->size; i++) {
        result->data[i] = main->data[i] - lambda;
    }
    #endif
    return result;
}
