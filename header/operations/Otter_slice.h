#ifndef OTTER_SLICE_H
#define OTTER_SLICE_H
#include <stdio.h>
#include <stdlib.h>
#include "../ottertensors.h"
#include "../ottertensors_utilities.h"
#include "../ottermath.h"
OtterTensor** OT_slice_tensor(OtterTensor* t, int channels, int kernel_size, int stride, int padding);



#endif // OTTER_SLICE_H