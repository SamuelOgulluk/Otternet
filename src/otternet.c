#include "../header/otternet.h"
#include "../header/otternet_optimizers.h"




Otternetwork* ON_initialise_otternetwork() { 
        Otternetwork* network = malloc(sizeof(Otternetwork));
    if (network == NULL) {
        fprintf(stderr, "Failed to allocate memory for network\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialiser tous les champs à zéro/NULL d'abord
    memset(network, 0, sizeof(Otternetwork));
    
    network->num_layers = 0;
    network->layers = NULL;
    network->end= NULL;
    network->start = NULL;
    network->error_function = NULL;
    network->optimizer = -1; // -1 means no optimizer set
    network->learning_rate = 0.01; // Default learning rate
    network->optimizer_params = NULL; // No optimizer parameters by default
    network->order = NULL; // No order set initially
    network->end_of_line = NULL;
    network->num_end_of_line = 0; // No layers with no forward connections
    network->start_of_line = NULL;
    network->num_start_of_line = 0; // No layers with no backward connections
    network->output = NULL; // No output tensor set initially
    network->errors = NULL; // No errors tensor set initially
    network->input = NULL; // No input tensor set initially
    return network;
}


void ON_add_layer(Otternetwork* network, Otterchain* new_layer) {
    if (network->layers == NULL) {
    network->layers = new_layer;
    network->start  = new_layer;
    } else {
        network->end->next = new_layer;
    }
    network->end = new_layer;
    network->num_layers++;
}

void ON_handle_optimizer_params(Otternetwork* network, char* optimizer, float* optimizer_params) {
    if (strcmp(optimizer, "SGD") == 0) {
        network->optimizer = 0;
        network->optimizer_params = NULL; // No parameters needed for SGD
    } else if (strcmp(optimizer, "SGDM") == 0) {
        network->optimizer = 1;
        network->optimizer_params = malloc(1 * sizeof(float));
        if(optimizer_params == NULL){
            network->optimizer_params[0] = 0.9f; // Default momentum
            printf("No momentum parameter provided. Using default: %.2f\n", network->optimizer_params[0]);
        }else{
            network->optimizer_params[0] = optimizer_params[0];  
        }
    } else if (strcmp(optimizer, "Adam") == 0) {
        network->optimizer = 2;
        network->optimizer_params = malloc(3 * sizeof(float));
        if(optimizer_params == NULL){
            network->optimizer_params[0] = 0.9f; // Default momentum
            network->optimizer_params[1] = 0.999f; // Default second moment
            network->optimizer_params[2] = 1e-8f; // Default epsilon
            printf("No Adam parameters provided. Using defaults: beta1=%.2f, beta2=%.3f, epsilon=%.1e\n", network->optimizer_params[0], network->optimizer_params[1], network->optimizer_params[2]);
        }else{
            network->optimizer_params[0] = optimizer_params[0];
            network->optimizer_params[1] = optimizer_params[1]; 
            network->optimizer_params[2] = optimizer_params[2]; 
        }
    } else {
        fprintf(stderr, "Unknown optimizer: %s\n", optimizer);
        exit(EXIT_FAILURE);
    }
}



void ON_compile_otternetwork(Otternetwork* network, char* optimizer, char* error_function, float learning_rate, float* optimizer_params) {
    network->error_function = strdup(error_function);
    network->learning_rate = learning_rate;
    ON_handle_optimizer_params(network, optimizer, optimizer_params);
    printf("Compiling network with %s optimizer and %s error function...\n", optimizer, error_function);
    network->order = calculate_distances_ordered(network);
    printf("Network order calculated.\n");
    Otterchain* current_layer = network->layers;
    network->start = current_layer;
    for(int i=0;i<network->num_layers;i++){
        for(int i = 0; i < current_layer->num_connections_backward; i++) {
            current_layer->connections_backward[i]->connections_forward = realloc(current_layer->connections_backward[i]->connections_forward, sizeof(Otterchain*) * (current_layer->connections_backward[i]->num_connections_forward + 1));
            current_layer->connections_backward[i]->connections_forward[current_layer->connections_backward[i]->num_connections_forward] = current_layer;
            current_layer->connections_backward[i]->num_connections_forward++;
        }
        switch(current_layer->type){
            case LAYER_DENSE: // Dense layer
                ON_compile_Dense_layer(current_layer);
                break;
            case LAYER_CONV1D: // Conv1D layer
                //ON_compile_Conv1D_layer(current_layer);
                break;
            case LAYER_FLATTEN: // Flatten layer
                //ON_compile_Flatten_layer(current_layer);
                break;
            default:
                fprintf(stderr, "Unknown layer type for compilation.\n");
                exit(EXIT_FAILURE);
        }
        current_layer = current_layer->next;
        if(i == network->num_layers - 1){
            network->end = current_layer;
        }
    }
    for (int i = 0; i < network->num_layers; i++) {
        Otterchain* node = network->order[i];
        node->network_rank = i;
        
        if (node->num_connections_backward == 0) {
            Otterchain** tmp = realloc(network->start_of_line, (network->num_start_of_line + 1) * sizeof(Otterchain*));
            if (!tmp) {
                fprintf(stderr, "Failed to realloc start_of_line\n");
                exit(EXIT_FAILURE);
            }
            network->start_of_line = tmp;
            network->start_of_line[network->num_start_of_line] = node;
            node->idx_input = network->num_start_of_line; 
            network->num_start_of_line++;
        }
        
        if (node->num_connections_forward == 0) {
            Otterchain** tmp = realloc(network->end_of_line, (network->num_end_of_line + 1) * sizeof(Otterchain*));
            if (!tmp) {
                fprintf(stderr, "Failed to realloc end_of_line\n");
                exit(EXIT_FAILURE);
            }
            network->end_of_line = tmp;
            network->end_of_line[network->num_end_of_line] = node;
            node->idx_output = network->num_end_of_line; 
            network->num_end_of_line++;
        }
    }


    network->errors = calloc(network->num_end_of_line, sizeof(OtterTensor*));
    network->input = calloc(network->num_start_of_line, sizeof(OtterTensor*));
    network->output = calloc(network->num_end_of_line,sizeof(OtterTensor*));
    

    printf("Network compiled correctly with %s optimizer and %s error function.\n", optimizer, error_function);
    return;
}





Otterchain** calculate_distances_ordered(Otternetwork* net) {
    int n = net->num_layers;

    // 1) Récupérer tous les nœuds dans un tableau
    Otterchain** nodes    = calloc(n, sizeof *nodes);
    if (nodes == NULL) {
        fprintf(stderr, "Failed to allocate memory for nodes\n");
        exit(EXIT_FAILURE);
    }
    int*        in_degree = calloc(n, sizeof *in_degree);
    if (in_degree == NULL) {
        fprintf(stderr, "Failed to allocate memory for in_degree\n");
        free(nodes);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        nodes[i] = (i == 0
            ? net->layers
            : nodes[i-1]->next
        );
    }

    // 2) Calculer in_degree (nombre d'arêtes entrantes) via connections_backward
    for (int i = 0; i < n; i++) {
        Otterchain* layer = nodes[i];
        for (int j = 0; j < layer->num_connections_backward; j++) {
            Otterchain* pred = layer->connections_backward[j];
            int idx = find_index(nodes, n, pred);
            if (idx >= 0) {
                // couche `pred` → couche `layer`
                in_degree[i]++;
            }
        }
    }

    // 3) Préparer la file (résultat servira aussi de file)
    Otterchain** result = malloc(n * sizeof *result);
    if (result == NULL) {
        fprintf(stderr, "Failed to allocate memory for result\n");
        free(nodes);
        free(in_degree);
        exit(EXIT_FAILURE);
    }
    int head = 0, tail = 0;
    // Enfiler les couches sans prédécesseur
    for (int i = 0; i < n; i++) {
        if (in_degree[i] == 0) {
            result[tail++] = nodes[i];
        }
    }

    // 4) Parcourir la file
    while (head < tail) {
        Otterchain* u = result[head++];
        // Pour chaque voisin v : c'est tout node v dont u est dans connections_backward
        for (int k = 0; k < n; k++) {
            Otterchain* v = nodes[k];
            // chercher si u ∈ v->connections_backward
            for (int m = 0; m < v->num_connections_backward; m++) {
                if (v->connections_backward[m] == u) {
                    if (--in_degree[k] == 0) {
                        result[tail++] = v;
                    }
                    break;
                }
            }
        }
    }

    free(nodes);
    free(in_degree);

    for(int i = 0; i< net->num_layers; i++) {
        result[i]->network_rank = i; 
    }

    return result;
}


OtterTensor** ON_feed_forward(Otternetwork* network, OtterTensor** input, int gradient_register) {
    for (int i = 0; i < network->num_start_of_line; i++) {
        
        free_malloc_tensor(&network->input[i]);
        network->input[i] = OT_copy(input[i]);
    }

    for (int i = 0; i < network->num_layers; i++) {
        switch (network->order[i]->type) {
            case LAYER_DENSE:
                ON_Dense_layer_forward(network, network->order[i]);
                break;
        
            default:
                fprintf(stderr, "Unknown layer type for feed forward.\n");
                exit(EXIT_FAILURE);
                break;
        }
    }
    OtterTensor** output = calloc(network->num_end_of_line, sizeof(OtterTensor*));
    if (output == NULL) {
        fprintf(stderr, "Failed to allocate memory for output tensors\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < network->num_end_of_line; i++) {
        free_malloc_tensor(&network->output[i]);

        network->output[i] = OT_copy(network->end_of_line[i]->post_activations);
        output[i] = OT_copy(network->output[i]);
    }
    if(!gradient_register){

        ON_reset_network(network);
    }



    return output;
}


OtterTensor* ON_Cost_derivative(OtterTensor* output, OtterTensor* labels, char* error_function){
    if (strcmp(error_function, "MSE") == 0) {
        OtterTensor* diff = OT_tensors_substract(output, labels);
        OT_ref_scalar_multiply(diff, 2.0f);
        return diff;
    }
    return NULL;
}



float ON_cost(OtterTensor* output, OtterTensor* labels, char* error_function){
    if (strcmp(error_function, "MSE") == 0) {
        OtterTensor* diff = OT_tensors_substract(output, labels);
        OT_ref_square(diff);
        float diff_sum = OT_sum(diff);
        free_malloc_tensor(&diff);
        return diff_sum;
    }
    else {
        fprintf(stderr, "Unknown error function: %s\n", error_function);
        exit(EXIT_FAILURE);
        return -1.0f; // Just to satisfy the compiler, this line will never be reached
    }
}



void ON_update_weights_and_biases(Otternetwork* network) {
    for (int i = 0; i < network->num_layers; i++) {
        for (int j = 0; j < network->order[i]->weights_depth; j++) {
            OT_ref_tensors_sum(network->order[i]->weights[j], network->order[i]->weights_gradients[j], "ON_update_weights_and_biases1");
            
            OT_ref_tensors_sum(network->order[i]->biases[j], network->order[i]->biases_gradients[j], "ON_update_weights_and_biases2");
            
        }

    }
}



OtterTensor** ON_predict(Otternetwork* network, OtterTensor** input) {
    OtterTensor** predictions = ON_feed_forward(network, input, 0);
    return predictions;
}



void ON_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size) {
    switch (network->optimizer) 
    {
    case 0:
        ON_SGD_fit(network, inputs, labels, epochs, batch_size);
        break;
    case 1:
        ON_SGDM_fit(network, inputs, labels, epochs, batch_size);
        break;
    case 2:
        ON_Adam_fit(network, inputs, labels, epochs, batch_size);
        break;
    default:
        fprintf(stderr, "Unknown optimizer: %d\n", network->optimizer);
        exit(EXIT_FAILURE);
        break;
    }
    return;
}


void ON_reset_network(Otternetwork* network) {
    if (!network) return;
    for(int i=0;i<network->num_layers;i++){
        switch (network->order[i]->type) {
            case LAYER_DENSE:
                ON_reset_layer(network->order[i]);
                break;
            default:
                fprintf(stderr, "Unknown layer type for feed forward.\n");
                exit(EXIT_FAILURE);
                break;
        }
    }
    for (int i = 0; i < network->num_start_of_line; i++) {
        free_malloc_tensor(&network->input[i]);
        network->input[i] = NULL;
    }
    for (int i = 0; i < network->num_end_of_line; i++) {
        free_malloc_tensor(&network->output[i]);
        network->output[i] = NULL;
    } 


}


void free_otternetwork(Otternetwork* network) {
    if (!network) return;
    


    for (int i = 0; i < network->num_layers; i++) {
        switch (network->order[i]->type) {
            case LAYER_DENSE:
                ON_free_layer(network->order[i]);
                
                break;
            default:
                fprintf(stderr, "Unknown layer type for feed forward.\n");
                exit(EXIT_FAILURE);
                break;
        }

    }
    for (int j = 0; j < network->num_start_of_line; j++) {
        free_malloc_tensor(&network->input[j]);
        network->input[j] = NULL;
    }


    // Libérer les entrées du réseau
    
    free_ottertensor_list(network->input, network->num_start_of_line);
    
    // Libérer les tableaux d'ordre et de connexions
    if (network->order) {
        free(network->order);
        network->order = NULL;
    }
    
    if (network->optimizer_params) {
        free(network->optimizer_params);
        network->optimizer_params = NULL;
    }
    
    if (network->error_function) {
        free(network->error_function);
        network->error_function = NULL;
    }
    
    if (network->start_of_line) {
        free(network->start_of_line);
        network->start_of_line = NULL;
    }
    
    if (network->end_of_line) {
        free(network->end_of_line);
        network->end_of_line = NULL;
    }
    

    free_ottertensor_list(network->output, network->num_end_of_line);
    

    free_ottertensor_list(network->errors, network->num_end_of_line);
    
    free(network);
    return;
}




void ON_reset_layer(Otterchain* chain) {
    if (chain->post_activations) {
        free_malloc_tensor(&chain->post_activations);
    }
    if (chain->local_errors) {
        free_malloc_tensor(&chain->local_errors);
    }
    if (chain->input) {
        for(int i=0;i<chain->num_connections_backward;i++){
            free_malloc_tensor(&chain->input[i]);
        }

    }
}

void ON_free_layer(Otterchain* layer){
    ON_reset_layer(layer);
    free_otterchain(layer);
}




void free_otterchain(Otterchain* chain) {
    if (!chain) return;
    
    if (chain->layer) {
        if (chain->type == LAYER_DENSE) {
            ON_free_Dense_layer((Dense_layer*)chain->layer);    
        } else if (chain->type == LAYER_CONV1D) {
            // free_Conv1D_layer((Conv1D_layer*)chain->layer);
        } else if (chain->type == LAYER_FLATTEN) {
            // free_Flatten_layer((Flatten_layer*)chain->layer);
        } else {
            fprintf(stderr, "Unknown layer type for freeing.\n");
            exit(EXIT_FAILURE);
        } 
    }

    if (chain->input_dims) {
        free(chain->input_dims);
        chain->input_dims = NULL;
    }
    if (chain->output_dims) {
        free(chain->output_dims);
        chain->output_dims = NULL;
    }


    free_ottertensor_list(chain->weights, chain->weights_depth);
    free_ottertensor_list(chain->biases, chain->weights_depth);
    free_ottertensor_list(chain->weights_gradients, chain->weights_depth);
    free_ottertensor_list(chain->biases_gradients, chain->weights_depth);
    
    
    if (chain->connections_backward) {
        free(chain->connections_backward);
        chain->connections_backward = NULL;
    }
    
    if (chain->connections_forward) {
        free(chain->connections_forward);
        chain->connections_forward = NULL;
    }



    free_malloc_tensor(&chain->local_errors);
    if(chain->input) {
        if(chain->num_connections_backward!=0) {
            free_ottertensor_list(chain->input, chain->num_connections_backward);
        }
        else {
            free_malloc_tensor(&chain->input[0]); 
            free(chain->input);
            chain->input = NULL;
        }
    }
    free_malloc_tensor(&chain->post_activations);

    
    free(chain);
    chain = NULL;
}

