#include "../header/ottertensors_operations.h"

void OT_ref_tensors_sum(OtterTensor* a, OtterTensor* b, const char* caller_name) {

    if (a->size != b->size) {
        fprintf(stderr, "[Error in %s] Tensor sizes do not match for addition: %d vs %d\n", caller_name, a->size, b->size);
        printf("Tensor A:\n");
        print_tensor(a,2);
        printf("Tensor B:\n");
        print_tensor(b,2);
        exit(EXIT_FAILURE);
    }

    if (a->rank != b->rank ) {
        fprintf(stderr, "[Error in %s] Tensors must have the same rank for addition. Found rank %i and %i\n", caller_name, a->rank, b->rank);
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->size; i++) {
        a->data[i] = a->data[i] + b->data[i];
    }
    return;
}

void OT_ref_tensors_substract(OtterTensor* a, OtterTensor* b) {

    if (a->size != b->size) {
        fprintf(stderr, "[Error in ] Tensor sizes do not match for addition: %d vs %d\n",  a->size, b->size);
        printf("Tensor A:\n");
        print_tensor(a,2);
        printf("Tensor B:\n");
        print_tensor(b,2);
        exit(EXIT_FAILURE);
    }

    if (a->rank != b->rank ) {
        fprintf(stderr, "Tensors must have the same rank for subtraction.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->size; i++) {
        a->data[i] = a->data[i] - b->data[i];
    }
    return;
}




OtterTensor* OT_tensors_sum(OtterTensor* a, OtterTensor* b) {
    OtterTensor* result=OT_zeros(a->dims, a->rank);

    if (a->rank != b->rank || a->rank != result->rank) {
        fprintf(stderr, "Tensors must have the same rank for addition.\n found rank %i and %i \n",a->rank,b->rank);
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

OtterTensor* OT_tensors_substract(OtterTensor* a, OtterTensor* b) {
    OtterTensor* result=OT_zeros(a->dims, a->rank);
    if (a->rank != b->rank || a->rank != result->rank) {
        fprintf(stderr, "Tensors must have the same rank for subtraction.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    return result;
}

OtterTensor* OT_Matrix_multiply(OtterTensor* a, OtterTensor* b) {
    if(a->rank>2 || b->rank>2 ){
        printf("matrix multiplcation is only possible for rank 2 and 1 matrices");
        exit(EXIT_FAILURE);
    }
    else if(a->rank==0){
        return(OT_scalar_multiply(b,a->data[0]));
    } else if (b->rank==0){
        return(OT_scalar_multiply(a,b->data[0]));
    }else if (a->dims[1] != b->dims[0]) {
        printf( "Inner dimensions must match for matrix multiplication.\n");
        exit(EXIT_FAILURE);
    }else {
        OtterTensor* result=OT_zeros((int[2]){a->dims[0],b->dims[1]},2);
        for(int i =0;i<a->dims[0];i++){
            for(int j = 0 ; j<b->dims[1];j++){
                for(int k = 0 ; k<a->dims[1];k++){
                    result->data[index_tensor(result,(int[2]){i,j})] += a->data[index_tensor(a,(int[2]){i,k})] * b->data[index_tensor(b,(int[2]){k,j})];
                }
            }
        }
        return result;
    }
}


OtterTensor* OT_dot(OtterTensor* a, OtterTensor* b) {
    OtterTensor* result = OT_zeros(a->dims, a->rank);
    if (a->rank != b->rank || a->rank != result->rank) {
        fprintf(stderr, "Tensors must have the same rank for multiplication.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}


void OT_ref_scalar_multiply(OtterTensor* main, float lambda) {
    for (int i = 0; i < main->size; i++) {
        main->data[i] *= lambda;
    }
    return;
}

OtterTensor* OT_scalar_multiply(OtterTensor* main, float lambda) {
    OtterTensor* result=OT_zeros(main->dims,main->rank);
    for (int i = 0; i < main->size; i++) {
        result->data[i] = lambda * main->data[i];
    }
    return result;
}

void OT_ref_dot_divide(OtterTensor* dividend, OtterTensor* divisor) {
    for (int i = 0; i < dividend->size; i++) {
        if (divisor->data[i] == 0.0f) {
            fprintf(stderr, "Error: Division by zero in OT_ref_dot_divide at index %d\n", i);
            dividend->data[i] = 0.0f; // or NAN
        } else {
            dividend->data[i] /= divisor->data[i];
        }
    }
    return;
}


OtterTensor* OT_dot_divide(OtterTensor* main, OtterTensor* divisor) {
    OtterTensor* result = OT_zeros(main->dims, main->rank);
    OT_ref_dot_divide(result, divisor);
    return result;
}

OtterTensor* OT_scalar_add(OtterTensor* main, float lambda) {
    OtterTensor* result=OT_zeros(main->dims, main->rank);
    for (int i = 0; i < main->size; i++) {
        result->data[i] = main->data[i] + lambda;
    }
    return result;
}

void OT_ref_scalar_sum(OtterTensor* main, float lambda) {
    for (int i = 0; i < main->size; i++) {
        main->data[i] += lambda;
    }
    return;
}

OtterTensor* OT_scalar_subtract(OtterTensor* main, float lambda) {
    OtterTensor* result=OT_zeros(main->dims,main->rank);
    for (int i = 0; i < main->size; i++) {
        result->data[i] = main->data[i] - lambda;
    }
    return result;
}


OtterTensor* OT_Transpose(OtterTensor* t) {
    if (t->rank != 2) {
        fprintf(stderr, "Transpose is only defined for 2D tensors.\n");
        exit(EXIT_FAILURE);
    }
    OtterTensor* transposed = OT_zeros((int[2]){t->dims[1], t->dims[0]}, 2);
    for (int i = 0; i < t->dims[0]; i++) {
        for (int j = 0; j < t->dims[1]; j++) {
            transposed->data[j * t->dims[0] + i] = t->data[i * t->dims[1] + j];
        }
    }
    return transposed;
}

void OT_ref_square(OtterTensor* t) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] *= t->data[i];
    }
    return;
}

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






OtterTensor* OT_column_sum(OtterTensor* t) {
    if (t->rank != 2) {
        fprintf(stderr, "Column sum is only defined for 2D tensors.\n");
        exit(EXIT_FAILURE);
    }
    
    OtterTensor* result = OT_zeros((int[2]){t->dims[1],1}, 2);
    for (int j = 0; j < t->dims[1]; j++) {
        for (int i = 0; i < t->dims[0]; i++) {
            result->data[j] += t->data[j + i * t->dims[1]];
        }
    }
    return result;
}

OtterTensor* OT_line_sum(OtterTensor* t) {
    if (t->rank != 2) {
        fprintf(stderr, "line sum is only defined for 2D tensors.\n");
        exit(EXIT_FAILURE);
    }
    
    OtterTensor* result = OT_zeros((int[2]){t->dims[0],1}, 2);
    for (int j = 0; j < t->dims[0]; j++) {
        for (int i = 0; i < t->dims[1]; i++) {
            result->data[j] += t->data[j*t->dims[1]+i];
        }
    }
    return result;
}

float OT_sum(OtterTensor* t) {
    float total = 0.0f;
    for (int i = 0; i < t->size; i++) {
        total += t->data[i];
    }
    return total;
}

void OT_ref_sqrt(OtterTensor* t) {
    for (int i = 0; i < t->size; i++) {
        if (t->data[i] < 0.0f) {
            fprintf(stderr, "Error: Cannot compute square root of a negative number in OT_ref_sqrt at index %d\n", i);
            exit(EXIT_FAILURE);
        }
        t->data[i] = OM_sqrt(t->data[i]);
    }
    return;
}




void OT_ref_reset(OtterTensor* t) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] = 0.0f;
    }
    return;
}

void OT_ref_copy(OtterTensor* dest, OtterTensor* src) {
    if (dest->size != src->size || dest->rank != src->rank) {
        fprintf(stderr, "Error: Tensor sizes or ranks do not match for copy operation.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < dest->size; i++) {
        dest->data[i] = src->data[i];
    }
    return;
}