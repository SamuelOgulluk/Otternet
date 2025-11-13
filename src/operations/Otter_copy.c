#include "../../header/operations/Otter_copy.h"

void OT_ref_copy(OtterTensor* dest, OtterTensor* src) {
    if (dest->size != src->size || dest->rank != src->rank) {
        fprintf(stderr, "Error: Tensor sizes or ranks do not match for copy operation.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < dest->size; i++) {
        dest->data[i] = src->data[i];
    }
    return;
}