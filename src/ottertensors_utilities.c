#include "../header/ottertensors_utilities.h"

void print_tensor_recursive(OtterTensor* t, int level, int ndims,int idx,int significant_digits) {
    if (level == ndims - 1) {
        printf("[");
        for (int i = 0; i < t->dims[level]; i++) {
            printf("%.*f",significant_digits,t->data[idx+i]);
            if (i != t->dims[level] - 1) {
                printf(",");
            }
        }
        printf("]");
    } else {
        // Dimensions intermédiaires : on ouvre une liste, on appelle récursivement
        printf("[");
        for (int i = 0; i < t->dims[level]; i++) {
            print_tensor_recursive(t, level + 1, ndims, idx + i * t->strides[level], significant_digits);
            if (i != t->dims[level] - 1) {
                printf(",");
            }
        }
        printf("]");
    }
}


void print_tensor(OtterTensor* t,int significant_digits) {
    /*printf("Ottertensor with shape: (");
    for (int i = 0; i < t->rank; i++) {
        printf("%d", t->dims[i]);
        if (i < t->rank - 1) {
            printf(", ");
        }
    }
    printf(")\n");
    */

    if (t->device == DEVICE_CUDA) {
        OT_to_cpu(t);
    }

    printf("Ottertensor(");
    if (t->rank ==0) {
        printf("%.*f",significant_digits,t->data[0]);
    } else {
        
        //printf("[");

        print_tensor_recursive(t, 0, t->rank, 0, significant_digits);
        //printf("]");
    }
    printf(")\n");
}


OtterTensor* OT_copy(OtterTensor* a){
    if(!a) return NULL;
    OtterTensor* result = OT_zeros(a->dims, a->rank);
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i];
    }
    return result;
}

OtterTensor** OT_copy_list(OtterTensor** a,int n){
    OtterTensor** result = malloc(n * sizeof(OtterTensor*));
    if (result == NULL) {
        fprintf(stderr, "Failed to allocate memory for tensor list\n");
        exit(EXIT_FAILURE);
    }
    for(int i=0;i<n;i++){
        result[i] = OT_copy(a[i]);
    }
    
    return result;
}

void OT_initialize_copy(OtterTensor* a, OtterTensor* copy){
    if (copy->dims) free(copy->dims);
    if (copy->strides) free(copy->strides);
    if (copy->data) free(copy->data);
    copy->dims = NULL;
    copy->strides = NULL;
    copy->data = NULL;
    set_dims(copy, a->dims, a->rank);
    copy->data = malloc(copy->size * sizeof(float));
    for (int i = 0; i < a->size; i++) {
        copy->data[i] = a->data[i];
    }
    return;
}


OtterTensor* OT_Flatten(OtterTensor* t) {
    OtterTensor* flat_tensor = OT_copy(t);
    for (int i = 0; i < t->size; i++) {
        flat_tensor->data[i] = t->data[i];
    }
    
    return flat_tensor;
}




OtterTensor* OT_zeros(int* dims, int rank){
    OtterTensor* tensor = malloc(sizeof(OtterTensor));
    if (tensor == NULL) {
        fprintf(stderr, "Failed to allocate memory for tensor\n");
        exit(EXIT_FAILURE);
    }
    set_dims(tensor, dims, rank);
    tensor->data = calloc(tensor->size, sizeof(float));
    if (tensor->data == NULL) {
        fprintf(stderr, "Failed to allocate memory for tensor data\n");
        free(tensor->dims);
        free(tensor->strides);
        free(tensor);
        exit(EXIT_FAILURE);
    }
    return tensor;
}

OtterTensor* OT_ones(int* dims, int rank){
    OtterTensor* tensor = malloc(sizeof(OtterTensor));
    if (tensor == NULL) {
        fprintf(stderr, "Failed to allocate memory for tensor\n");
        exit(EXIT_FAILURE);
    }
    set_dims(tensor, dims, rank);
    tensor->data = malloc(tensor->size * sizeof(float));
    if (tensor->data == NULL) {
        fprintf(stderr, "Failed to allocate memory for tensor data\n");
        free(tensor->dims);
        free(tensor->strides);
        free(tensor);
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < tensor->size; i++) {
        tensor->data[i] = 1.0f;
    }
    return tensor;
}

OtterTensor*** OT_tensor_duplicate(OtterTensor** tensors, int size_tensor, int n) {
    OtterTensor*** result = malloc(n * sizeof(OtterTensor**));
    if (result == NULL) {
        fprintf(stderr, "Failed to allocate memory for tensor duplicate\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        result[i] = OT_copy_list(tensors,size_tensor);
    }
    return result;
}


Otterlist* OT_otterlist(OtterTensor** tensor_list, int size) {
    Otterlist* list = malloc(sizeof(Otterlist));
    if (list == NULL) {
        fprintf(stderr, "Failed to allocate memory for otterlist\n");
        exit(EXIT_FAILURE);
    }
    list->dataset = OT_copy_list(tensor_list, size);
    list->size = size;
    return list;
}

Otterlist* OT_init_otterlist(int size) {
    Otterlist* list = malloc(sizeof(Otterlist));
    if (list == NULL) {
        fprintf(stderr, "Failed to allocate memory for otterlist\n");
        exit(EXIT_FAILURE);
    }
    list->dataset = calloc(size, sizeof(OtterTensor*));
    if (list->dataset == NULL) {
        fprintf(stderr, "Failed to allocate memory for otterlist dataset\n");
        free(list);
        exit(EXIT_FAILURE);
    }
    list->size = size;
    return list;
}

void OT_free_otterlist(Otterlist* list) {
    if (!list) return;
    
    for (int i = 0; i < list->size; i++) {
        free_malloc_tensor(&list->dataset[i]);
    }
    
    free(list->dataset);
    list->dataset = NULL;
    free(list);
    list = NULL;
}