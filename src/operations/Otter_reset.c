#include "../../header/operations/Otter_reset.h"
void OT_ref_reset(OtterTensor* t) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] = 0.0f;
    }
    return;
}
