#ifndef OTTER_SUM_H
#define OTTER_SUM_H
#include <stdio.h>
#include <stdlib.h>
#include "../ottertensors.h"
#include "../ottertensors_utilities.h"
OtterTensor* OT_tensors_sum(OtterTensor* a, OtterTensor* b);
void OT_ref_tensors_sum(OtterTensor* a, OtterTensor* b, const char* caller_name);


#endif // OTTER_SUM_H