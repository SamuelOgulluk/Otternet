#include "../../header/operations/Otter_Tensor_square.h"

void OT_ref_square(OtterTensor* t) {
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < t->size; i++) {
        t->data[i] *= t->data[i];
    }
    #else
    for (int i = 0; i < t->size; i++) {
        t->data[i] *= t->data[i];
    }
    #endif
    return;
}