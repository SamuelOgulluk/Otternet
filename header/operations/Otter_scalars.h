#ifndef OTTER_SCALARS_H
#define OTTER_SCALARS_H
#include <stdio.h>
#include <stdlib.h>
#include "../ottertensors.h"
#include "../ottertensors_utilities.h"

void OT_ref_scalar_multiply(OtterTensor* main, float lambda);
OtterTensor* OT_scalar_multiply(OtterTensor* main, float lambda);
OtterTensor* OT_scalar_add(OtterTensor* main, float lambda);
void OT_ref_scalar_sum(OtterTensor* main, float lambda);
OtterTensor* OT_scalar_subtract(OtterTensor* main, float lambda);


#endif // OTTER_SCALARS_H