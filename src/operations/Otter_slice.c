#include "../../header/operations/Otter_slice.h"

OtterTensor** OT_slice_tensor(OtterTensor* t, int channels,int kernel_size, int stride, int padding) {
    // takes a tensor of rank 2 and returns a list of tensors of rank 2
    // each tensor is a slice of the original tensor with the given filter size and stride
    int input_length = t->dims[0];
    int pad = 0;
    int n_output_size;
    
    if(padding == 0){
        //padding valid
        pad = 0;
        n_output_size = (int)((input_length - kernel_size) / stride) + 1;
    } else if(padding ==1){
        //padding same
        n_output_size =(int)(input_length / stride);
        if(stride ==1){
            pad = (int)((kernel_size - 1)/2);
        } else {
            pad = (int)(OM_intmax((n_output_size*stride + kernel_size-input_length), 0)/2);
        }
    } else if (padding == 2) { //
        // padding full
        pad = kernel_size - 1;
        n_output_size = (int)((input_length -1) / stride) + kernel_size;
    } else if (padding != 0) {
        fprintf(stderr, "Invalid padding value: %d. Use 0, 1, 2, or 3.\n", padding);
        exit(EXIT_FAILURE);
    }

    
    OtterTensor** slices = malloc(channels * sizeof(OtterTensor*));
    printf("pad %i",pad);
    OtterTensor* padded_tensor = OT_zeros((int[]){input_length + 2 * pad,1}, 2);
    for (int i = 0; i < input_length; i++) {
        padded_tensor->data[i + pad] = t->data[i];
    }

    OtterTensor* slice = OT_zeros((int[]){n_output_size,kernel_size}, 2);
    for (int i = 0; i < n_output_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int index = i * stride + j;
            if (index < padded_tensor->dims[0]) {
                slice->data[i * kernel_size + j] = padded_tensor->data[index];
            } else {
                slice->data[i * kernel_size + j] = 0.0f;  // ou tout autre remplissage de ton choix
            }
        }
    }
    for(int i = 0; i < channels; i++) {
        slices[i] = OT_copy(slice);
    }
    free_malloc_tensor(&slice);
    free_malloc_tensor(&padded_tensor);
    
    return slices;
}
