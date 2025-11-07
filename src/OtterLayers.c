#include "../header/OtterLayers.h"

const char* LAYER_TYPE[] = {
    "Dense",
    "Flatten",
    "Conv1D",
    "Conv2D",
    "Conv3D",
    "MaxPooling",
    "AveragePooling",
    "Dropout",
    "BatchNormalization",
    "Recurrent",
    "LSTM",
    "GRU",
    "Transformer",
    "Attention",
    "Embedding",
};

Otterchain* ON_Dense_layer(int neurons, char* activation_function,Otterchain* previous_layer,int number_of_previous_layers,int input_size) {
    Otterchain* chain = malloc(sizeof(Otterchain));
    if (chain == NULL) {
        fprintf(stderr, "Failed to allocate memory for chain\n");
        exit(EXIT_FAILURE);
    }
    memset(chain, 0, sizeof(Otterchain));
    Dense_layer* layer = malloc(sizeof(Dense_layer));
    if (layer == NULL) {
        fprintf(stderr, "Failed to allocate memory for dense layer\n");
        free(chain);
        exit(EXIT_FAILURE);
    }
    memset(layer, 0, sizeof(Dense_layer));
    chain->weights = malloc(sizeof(OtterTensor*));
    if (chain->weights == NULL) {
        fprintf(stderr, "Failed to allocate memory for weights\n");
        free(layer);
        free(chain);
        exit(EXIT_FAILURE);
    }
    chain->biases = malloc(sizeof(OtterTensor*));
    if (chain->biases == NULL) {
        fprintf(stderr, "Failed to allocate memory for biases\n");
        free(chain->weights);
        free(layer);
        free(chain);
        exit(EXIT_FAILURE);
    }
    chain->weights_depth = 1;
    chain->layer = layer;
    layer->num_neurons = neurons;
    layer->activation_function = strdup(activation_function);
    if(number_of_previous_layers ==0){
        chain->num_connections_backward = 0;
        chain->connections_backward = NULL;
    } else{
        chain->connections_backward = malloc(sizeof(Otterchain*));
        if (chain->connections_backward == NULL) {
            fprintf(stderr, "Failed to allocate memory for backward connections\n");
            free(chain->biases);
            free(chain->weights);
            free(layer->activation_function);
            free(layer);
            free(chain);
            exit(EXIT_FAILURE);
        }
        chain->num_connections_backward = 1;
        chain->connections_backward[0] = previous_layer;
    }
    chain->next= NULL;
    chain->connections_forward = NULL;
    chain->num_connections_forward = 0;
    chain->input = NULL;
    if(input_size!=0){
        chain->input_dims = malloc(2 * sizeof(int));
        if (chain->input_dims == NULL) {
            fprintf(stderr, "Failed to allocate memory for input dimensions\n");
            // Free previously allocated memory
            if (chain->connections_backward) free(chain->connections_backward);
            free(chain->biases);
            free(chain->weights);
            free(layer->activation_function);
            free(layer);
            free(chain);
            exit(EXIT_FAILURE);
        }
        chain->input_dims[0] = input_size;
        chain->input_dims[1] = 1;
    } else {
        chain->input_dims = NULL;
    }
    chain->output_dims = NULL;
    chain->local_errors = NULL;
    chain->post_activations = NULL;
    chain->type = LAYER_DENSE; 

    return chain;
}




void ON_compile_Dense_layer(Otterchain* current_chain) { // faut gérer les connections avant et arriere
    
    int input_dims = 0;
    if (current_chain->num_connections_backward > 1) {
        fprintf(stderr, "Dense layer compilation : Dense can only take one input connection, received %i\n", current_chain->num_connections_backward);
        exit(EXIT_FAILURE);
    } 

    Otterchain** previous_layer= current_chain->connections_backward;

    if(current_chain->num_connections_backward == 0){
        if(!current_chain->input_dims){
            input_dims = ((Dense_layer*)current_chain->layer)->num_neurons; // If no previous layer, and no defined shape, use the number of neurons    
        }
        else{
            input_dims = current_chain->input_dims[0]; // If no previous layer, use the defined shape
        }
    } else if (previous_layer[0]->type == LAYER_DENSE) {
        input_dims = ((Dense_layer*)previous_layer[0]->layer)->num_neurons; // Dense layer
        
    } else if (previous_layer[0]->type == LAYER_FLATTEN) {
        input_dims = ((Flatten_layer*)previous_layer[0]->layer)->output_size; // Flatten layer
    } else {
        fprintf(stderr, "Unknown layer type for Dense layer compilation.\n");
        exit(EXIT_FAILURE);
    }


    int dims_w[2] = {((Dense_layer*)current_chain->layer)->num_neurons, input_dims};
    int dims_b[2] = {((Dense_layer*)current_chain->layer)->num_neurons, 1};
    current_chain->weights[0] = OT_random_uniform(dims_w, 2, -1.0f, 1.0f);
    current_chain->biases[0] = OT_random_uniform(dims_b, 2, -1.0f, 1.0f);

    if(!current_chain->input_dims){
        current_chain->input_dims = malloc(2 * sizeof(int));
        if (current_chain->input_dims == NULL) {
            fprintf(stderr, "Failed to allocate memory for input dimensions\n");
            exit(EXIT_FAILURE);
        }
        current_chain->input_dims[0] = input_dims;
        current_chain->input_dims[1] = 1;
        
    }
    current_chain->output_dims = malloc(2 * sizeof(int));
    if (current_chain->output_dims == NULL) {
        fprintf(stderr, "Failed to allocate memory for output dimensions\n");
        exit(EXIT_FAILURE);
    }
    current_chain->output_dims[0] = ((Dense_layer*)current_chain->layer)->num_neurons;
    current_chain->output_dims[1] = 1;
    
    current_chain->weights_depth= 1;

    current_chain->weights_gradients = calloc(1,sizeof(OtterTensor*));
    if (current_chain->weights_gradients == NULL) {
        fprintf(stderr, "Failed to allocate memory for weight gradients\n");
        exit(EXIT_FAILURE);
    }
    current_chain->biases_gradients = calloc(1,sizeof(OtterTensor*));
    if (current_chain->biases_gradients == NULL) {
        fprintf(stderr, "Failed to allocate memory for bias gradients\n");
        exit(EXIT_FAILURE);
    }
    current_chain->weights_gradients[0] = OT_zeros(dims_w, 2);
    current_chain->biases_gradients[0] = OT_zeros(dims_b, 2);

    current_chain->input = calloc(1,sizeof(OtterTensor*));
    if (current_chain->input == NULL) {
        fprintf(stderr, "Failed to allocate memory for input tensor\n");
        exit(EXIT_FAILURE);
    }
    current_chain->post_activations = NULL;
}



void ON_Dense_layer_forward(Otternetwork* net,Otterchain* chain) {
    
    OtterTensor* input = NULL;
    if (chain->num_connections_backward == 0) {
        input = OT_copy(net->input[chain->idx_input]); // If no previous layer, use the network input

    } else{
        input = OT_copy(chain->connections_backward[0]->post_activations);
    }

    if (!input) {
        fprintf(stderr, "Error: Dense layer received NULL input tensor.\n");
        exit(EXIT_FAILURE);
    }

    free_malloc_tensor(&chain->input[0]);
    chain->input[0] = input;
    

    OtterTensor* prod = OT_Matrix_multiply(chain->weights[0], input);
    CHECK_NAN_TENSOR(prod, "OT_Matrix_multiply layer X");
    OT_ref_tensors_sum(prod, chain->biases[0], "OT_Matrix_multiply layer 1X");
    CHECK_NAN_TENSOR(prod, "OT_Matrix_multiply layer 2X");
    Activation_functions(((Dense_layer*)chain->layer)->activation_function, prod);
    CHECK_NAN_TENSOR(prod, "OT_Matrix_multiply layer 3X");

     
    if (chain->post_activations) free_malloc_tensor(&chain->post_activations);
    chain->post_activations =prod;

    return;
}



OtterTensor* ON_Dense_layer_backward(Otternetwork* network, Otterchain* chain) {
    OtterTensor* error = NULL;
    if (chain->num_connections_forward == 0) {
        error = OT_copy(network->errors[chain->idx_output]);
        CHECK_NAN_TENSOR(error, "OT_Matrix_multiply layer 12X");
    } else {
        for (int i = 0; i < chain->num_connections_forward; ++i) {
            Otterchain* f = chain->connections_forward[i];
            if (!error) {
                error = OT_copy(f->local_errors);
                CHECK_NAN_TENSOR(error, "OT_Matrix_multiply layer 13");
            } else {
                OT_ref_tensors_sum(error, f->local_errors, "OT_Matrix_multiply layer 14");
                CHECK_NAN_TENSOR(error, "OT_Matrix_multiply layer 14");
            }
        }
    }

    if (!error) {
        fprintf(stderr, "Error: Null tensor encountered during backward pass.\n");
        exit(EXIT_FAILURE);
    }

    OtterTensor* dAct = OT_copy(chain->post_activations);
    CHECK_NAN_TENSOR(dAct, "OT_copy layer 10X");
    derivative_activation_functions(((Dense_layer*)chain->layer)->activation_function, dAct);
    CHECK_NAN_TENSOR(dAct, "derivative_activation_functions layer 11X");
    OtterTensor* dZ = OT_dot(error, dAct);

    CHECK_NAN_TENSOR(dZ, "O_Matrix_multiply layer 4X");
    free_malloc_tensor(&error);
    free_malloc_tensor(&dAct);

    OtterTensor* W_T = OT_Transpose(chain->weights[0]);
    CHECK_NAN_TENSOR(W_T, "OT_Transpose layer 9X");
    if (chain->local_errors) {
        free_malloc_tensor(&chain->local_errors);
        chain->local_errors = NULL; // Avoid dangling pointer
    }
    chain->local_errors = OT_Matrix_multiply(W_T, dZ);
    CHECK_NAN_TENSOR(chain->local_errors, "OT_Matrix_multiply layer 5X");
    free_malloc_tensor(&W_T);

    OtterTensor* X_T = OT_Transpose(chain->input[0]);
    CHECK_NAN_TENSOR(X_T, "OT_Transpose layer 8X");
    if (chain->weights_gradients[0]) {
        free_malloc_tensor(&chain->weights_gradients[0]);
        chain->weights_gradients[0] = NULL; // Avoid dangling pointer
    }
    chain->weights_gradients[0] = OT_Matrix_multiply(dZ, X_T);
    CHECK_NAN_TENSOR(chain->weights_gradients[0], "OT_Matrix_multiply layer 7X");
    free_malloc_tensor(&X_T);

    if (chain->biases_gradients[0]) {
        free_malloc_tensor(&chain->biases_gradients[0]);
        chain->biases_gradients[0] = NULL; // Avoid dangling pointer
    }
    /*     printf("dZ: \n");
    print_tensor(dZ, 2);
    printf("final dZ\n"); */
    chain->biases_gradients[0] = OT_line_sum(dZ);
    CHECK_NAN_TENSOR(chain->biases_gradients[0], "OT_Matrix_multiply layer 6X");
    free_malloc_tensor(&dZ);

    return chain->local_errors;
}



void ON_reset_Dense_layer(Otterchain* chain) {
    if (chain->post_activations) {
        free_malloc_tensor(&chain->post_activations);
    }
    if (chain->local_errors) {
        free_malloc_tensor(&chain->local_errors);
    }
    if (chain->input) {
        free_malloc_tensor(&chain->input[0]);

    }
}



void ON_free_Dense_layer(Dense_layer* layer) {
    if (!layer) return;
    
    if (layer->activation_function) {
        free(layer->activation_function);
    }
    free(layer);
}




//////////////////////////////////////////////////





Otterchain* ON_Conv1D_layer(int kernet_size,int filter, int stride, int padding, int neurons, char* activation_function) {
    Otterchain* chain = malloc(sizeof(Otterchain));
    if (chain == NULL) {
        fprintf(stderr, "Failed to allocate memory for chain\n");
        exit(EXIT_FAILURE);
    }
    Conv1D_layer* layer = malloc(sizeof(Conv1D_layer));
    if (layer == NULL) {
        fprintf(stderr, "Failed to allocate memory for Conv1D layer\n");
        free(chain);
        exit(EXIT_FAILURE);
    }
    layer->filter = filter;
    layer->kernel_size = kernet_size;
    layer->stride = stride;
    layer->padding = padding;
    chain->weights = malloc(filter * sizeof(OtterTensor*));
    if (chain->weights == NULL) {
        fprintf(stderr, "Failed to allocate memory for weights\n");
        free(layer);
        free(chain);
        exit(EXIT_FAILURE);
    }
    chain->biases = NULL;
    layer->num_neurons = neurons;
    layer->activation_function = strdup(activation_function);
    chain->type = LAYER_CONV1D; 
    chain->layer = layer;
    return chain;
}

void ON_compile_Conv1D_layer(Otterchain* chain, int input_length) {
    (void)input_length;
    (void)chain;
    /*
    for(int i = 0; i < ((Conv1D_layer*)chain->layer)->filter; i++) {
        ((Conv1D_layer*)chain->layer)->weights[i] = OT_random_uniform((int[2]){1,((Conv1D_layer*)chain->layer)->kernel_size}, 2, -1.0f, 1.0f);
    }
    ((Conv1D_layer*)chain->layer)->biases = OT_zeros((int[2]){((Conv1D_layer*)chain->layer)->kernel_size,1}, 2);
    
    chain->weights_depth = ((Conv1D_layer*)chain->layer)->filter;
    chain->input_dims = malloc(2 * sizeof(int));
    chain->output_dims = malloc(2 * sizeof(int));
    chain->input_dims[0] = input_length;
    chain->input_dims[1] = 1;

    */
}

/*
OtterTensor* ON_Conv1D_layer_forward(Otterchain* chain,OtterTensor* input, int gradient_register) { //à travailler
    
    int N=0;
    OtterTensor* prod = OT_zeros((int[2]){}, 2);
    OtterTensor** slice=OT_slice_padding(input, layer->filter, layer->stride, layer->padding);
    
    if(layer->padding){
        N = input->dims[0];
    }else{
        N= input->dims[0] - layer->filter + 1;
    }

    
    return prod;
}

void free_Conv1D_layer(Conv1D_layer* layer) {
    if (layer == NULL) return;
    for (int i = 0; i < layer->filter; i++) {
        free_malloc_tensor(layer->weights[i]);
    }
    free(layer->weights);
    free_malloc_tensor(layer->biases);
    if (layer->activation_function) {
        free(layer->activation_function);
    }
    free(layer);
}


/////////////////////////////////////////////////
*/


/*
Flatten_layer* ON_Flatten_layer(int neurons, char* activation_function) {
    Flatten_layer* layer = malloc(sizeof(ON_Flatten_layer));
    return layer;
}

void ON_compile_Flatten_layer(Otterchain* layer) {
    switch(layer->connections[0]->type) {
        case LAYER_DENSE: // Dense layer
            ((Flatten_layer*)layer->layer)->output_size = ((Dense_layer*)layer->connections[0]->layer)->num_neurons;
            break;
        case LAYER_CONV1D : 
            ((Flatten_layer*)layer->layer)->output_size = ((Conv1D_layer*)layer->connections[0]->layer)->output_dims[0]* ((Conv1D_layer*)layer->connections[0]->layer)->output_dims[1];
            break;
        case LAYER_FLATTEN: // Flatten layer
            ((Flatten_layer*)layer->layer)->output_size = ((Flatten_layer*)layer->connections[0]->layer)->output_size;
            break;
        default:
            fprintf(stderr, "Unknown layer type for Flatten layer compilation.\n");
            exit(EXIT_FAILURE);
    }
    return;
}
*/