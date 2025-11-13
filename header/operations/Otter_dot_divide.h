#ifndef OTTER_DOT_DIVIDE_H
#define OTTER_DOT_DIVIDE_H
#include <stdio.h>
#include <stdlib.h>
#include "../ottertensors.h"
#include "../ottertensors_utilities.h"
void OT_ref_dot_divide(OtterTensor* dividend, OtterTensor* divisor);
OtterTensor* OT_dot_divide(OtterTensor* main, OtterTensor* divisor);



#endif // OTTER_DOT_DIVIDE_H