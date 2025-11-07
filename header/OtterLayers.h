#ifndef OTTERLAYERS_H
#define OTTERLAYERS_H
#include "otternet.h"
#include "ottertensors.h"
#include "ottertensors_utilities.h"
#include "ottertensors_operations.h"
#include "OtterActivation.h"
#include "ottertensors_random.h"
#include "ottermath.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>



typedef enum {
    LAYER_DENSE,
    LAYER_FLATTEN,
    LAYER_CONV1D,
    LAYER_CONV2D,
    LAYER_CONV3D,
    LAYER_MAXPOOLING,
    LAYER_AVGPOOLING,
    LAYER_DROPOUT,
    LAYER_BATCHNORM,
    LAYER_RECURRENT,
    LAYER_LSTM,
    LAYER_GRU,
    LAYER_TRANSFORMER,
    LAYER_ATTENTION,
    LAYER_EMBEDDING
} LayerType;

extern const char* LAYER_TYPE[]; // Declare as extern, do not define here

typedef struct Otternetwork Otternetwork;

typedef struct Otterchain Otterchain;

struct Otterchain {
    Otterchain* next;
    void* layer;
    LayerType type; 
    
    Otterchain** connections_backward; 
    Otterchain** connections_forward; 
    int num_connections_backward; 
    int num_connections_forward; 

    OtterTensor** biases;
    OtterTensor** weights;

    OtterTensor** weights_gradients;
    OtterTensor** biases_gradients ;
    OtterTensor* local_errors;
    OtterTensor** input;             
    OtterTensor* post_activations;

    int weights_depth;
    int* input_dims; 
    int* output_dims; 
    int network_rank;
    
    int idx_output;
    int idx_input; // Index of the input layer in the network
};

//////////////////////////////////////////////////////////////////////

typedef struct Dense_layer {
    int num_neurons;
    char* activation_function;
    
}Dense_layer;


Otterchain* ON_Dense_layer(int neurons, char* activation_function,Otterchain* previous_layer,int number_of_previous_layers,int input_size);
void ON_compile_Dense_layer(Otterchain* layer);
void ON_Dense_layer_forward(Otternetwork* net,Otterchain* chain) ;
OtterTensor* ON_Dense_layer_backward(Otternetwork* network,Otterchain* chain) ;

void ON_free_Dense_layer(Dense_layer* layer);


//////////////////////////////////////////////////////


typedef struct Conv1D_layer{
    int kernel_size; 
    int filter;
    int stride;
    int padding;
    int num_neurons;
    char* activation_function;
    int* input_dims[2]; // Dimensions of the input tensor
    int* output_dims[2]; // Dimensions of the output tensor
    OtterTensor* output; // Output tensor after convolution
    
} Conv1D_layer;
/*
Otterchain* ON_Conv1D_layer(int kernel_size, int filter, int stride, int padding, int neurons, char* activation_function);
void ON_compile_Conv1D_layer(Otterchain* layer, int input_dims);
OtterTensor* ON_Conv1D_layer_forward(Conv1D_layer* layer, OtterTensor* input, OtterTensor** zs, OtterTensor** activations);
OtterTensor* ON_Conv1D_layer_backward(Otternetwork* network, Otterchain* chain, OtterTensor* input, int layer_number);
void free_Conv1D_layer(Conv1D_layer* layer);
*/
//////////////////////////////////////////////////////:


typedef struct Flatten_layer {
    int output_size;
} Flatten_layer;
/*
Flatten_layer* ON_Flatten_layer(int neurons, char* activation_function);
void ON_compile_Flatten_layer(Otterchain* layer);
OtterTensor* ON_Flatten_layer_forward(Dense_layer* layer, OtterTensor* input, OtterTensor** zs, OtterTensor** activations);
OtterTensor* ON_Flatten_layer_backward(Otternetwork* network, Otterchain* chain, OtterTensor* input, int layer_number);
void free_Flatten_layer(Flatten_layer* layer);
*/











#endif