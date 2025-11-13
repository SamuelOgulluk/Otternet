#ifndef OTTER_SUB_H
#define OTTER_SUB_H
#include <stdio.h>
#include <stdlib.h>
#include "../ottertensors.h"
#include "../ottertensors_utilities.h"

OtterTensor* OT_tensors_substract(OtterTensor* a, OtterTensor* b);
void OT_ref_tensors_substract(OtterTensor* a, OtterTensor* b);


#endif // OTTER_SUB_H