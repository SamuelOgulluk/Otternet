#ifndef OTTER_TENSORS_OPERATIONS_H
#define OTTER_TENSORS_OPERATIONS_H
#include "ottertensors.h"
#include "ottertensors_utilities.h"
#include "ottermath.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


OtterTensor* OT_tensors_sum(OtterTensor* a, OtterTensor* b);
void OT_ref_tensors_sum(OtterTensor* a, OtterTensor* b, const char* caller_name);
void OT_ref_tensors_substract(OtterTensor* a, OtterTensor* b);
OtterTensor* OT_tensors_substract(OtterTensor* a, OtterTensor* b);
OtterTensor* OT_Matrix_multiply(OtterTensor* a, OtterTensor* b);
OtterTensor* OT_dot(OtterTensor* a, OtterTensor* b);
void OT_ref_scalar_multiply(OtterTensor* main, float lambda);
OtterTensor* OT_scalar_add(OtterTensor* main, float lambda);
OtterTensor* OT_scalar_subtract(OtterTensor* main, float lambda);
OtterTensor* OT_scalar_multiply(OtterTensor* main, float lambda);
OtterTensor* OT_Transpose(OtterTensor* t);
void OT_ref_square(OtterTensor* t);
void OT_ref_scalar_sum(OtterTensor* main, float lambda);
OtterTensor* OT_dot_divide(OtterTensor* main, OtterTensor* divisor);
void OT_ref_dot_divide(OtterTensor* dividend, OtterTensor* divisor);
OtterTensor** OT_slice_tensor(OtterTensor* t, int channels, int kernel_size, int stride, int padding);
OtterTensor* OT_column_sum(OtterTensor* t);
OtterTensor* OT_line_sum(OtterTensor* t);
void OT_ref_reset(OtterTensor* t);
float OT_sum(OtterTensor* t);
void OT_ref_sqrt(OtterTensor* t);
void OT_ref_copy(OtterTensor* dest, OtterTensor* src) ;
#endif