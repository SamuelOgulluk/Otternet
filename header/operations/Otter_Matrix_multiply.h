#ifndef OTTERMATRIX_MULTIPLY_H
#define OTTERMATRIX_MULTIPLY_H
#include <stdio.h>
#include <stdlib.h>
#include "../ottertensors.h"
#include "../ottertensors_utilities.h"
OtterTensor* OT_Matrix_multiply(OtterTensor* a, OtterTensor* b);
void OT_Matrix_multiply_cpu(OtterTensor* a, OtterTensor* b, OtterTensor* result);
void OT_Matrix_multiply_cuda(OtterTensor* a, OtterTensor* b, OtterTensor* result);


#ifdef _OPENMP
    #include <omp.h>
#endif

#ifdef USE_CUDA
    #include <cuda_runtime.h>
    void launch_matmul_kernel(float* d_a, float* d_b, float* d_res, int M, int K, int N);
#endif



#endif // OTTERMATRIX_MULTIPLY_H