#ifndef OTTERCUDA_H
#define OTTERCUDA_H

#include "ottertensors.h"
#include "ottertensors_utilities.h"

void OTC_init();
void OT_to_cpu(OtterTensor* t);
void OT_to_cuda(OtterTensor* t);




#endif // OTTERCUDA_H