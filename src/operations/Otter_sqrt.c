#include "../../header/operations/Otter_sqrt.h"
void OT_ref_sqrt(OtterTensor* t) {
    for (int i = 0; i < t->size; i++) {
        if (t->data[i] < 0.0f) {
            fprintf(stderr, "Error: Cannot compute square root of a negative number in OT_ref_sqrt at index %d\n", i);
            exit(EXIT_FAILURE);
        }
        t->data[i] = OM_sqrt(t->data[i]);
    }
    return;
}
