#ifndef Otter_AXES_SUMS_H
#define Otter_AXES_SUMS_H
#include <stdio.h>
#include <stdlib.h>
#include "../ottertensors.h"
#include "../ottertensors_utilities.h"


OtterTensor* OT_column_sum(OtterTensor* t);
OtterTensor* OT_line_sum(OtterTensor* t);
float OT_sum(OtterTensor* t);




#endif // Otter_AXES_SUMS_H