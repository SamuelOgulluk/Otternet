#include "../header/otternet_optimizers.h"


OtterTensor**** ON_init_clone_weights(Otternetwork* network) {
    OtterTensor*** accum_weights = calloc(network->num_layers, sizeof(OtterTensor**));
    OtterTensor*** accum_biases  = calloc(network->num_layers, sizeof(OtterTensor**));
    for (int l = 0; l < network->num_layers; l++) {
        int depth = network->order[l]->weights_depth;
        accum_weights[l] = calloc(depth, sizeof(OtterTensor*));
        accum_biases[l]  = calloc(depth, sizeof(OtterTensor*));
        for (int k = 0; k < depth; k++) {
            /* on clone la forme des poids/bias afin d'initialiser à zéro */
            accum_weights[l][k] = OT_zeros(network->order[l]->weights[k]->dims, network->order[l]->weights[k]->rank);
            accum_biases[l][k]  = OT_zeros(network->order[l]->biases[k]->dims, network->order[l]->biases[k]->rank);
            
        }
    }
    OtterTensor**** result = malloc(2 * sizeof(OtterTensor***));
    result[0] = accum_weights;
    result[1] = accum_biases;
    return result;
}

void ON_reset_clone_weight(Otternetwork* net, OtterTensor*** accu){
    for(int i=0;i<net->num_layers;i++){
        for(int j=0;j<net->order[i]->weights_depth;j++){
            OT_ref_reset(accu[i][j]);
        }
    }
}

void free_clone_weights(Otternetwork* network, OtterTensor*** accu) {
    for (int l = 0; l < network->num_layers; l++) {
        int depth = network->order[l]->weights_depth;
        for (int k = 0; k < depth; k++) {
            free_malloc_tensor(&accu[l][k]);
        }
        free(accu[l]);
        accu[l] = NULL;
    }
    free(accu);
    accu = NULL;
}


void ON_SGD_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size) {
    OtterTensor**** accumulators = ON_init_clone_weights(network);
    OtterTensor*** accum_weights = accumulators[0];
    OtterTensor*** accum_biases  = accumulators[1];
    float inv_batch_size = 1.0f / (float)batch_size;
    int step = epochs / 9;
    for (int epoch = 0; epoch < epochs; epoch++) {
        int* indices = OR_select_batch(inputs->size[0], batch_size);

        for (int i = 0; i < batch_size; i++) {
            int idx = indices[i];
            OtterTensor** input_arr = calloc(network->num_start_of_line, sizeof(OtterTensor*));
            OtterTensor** label_arr = calloc(network->num_end_of_line, sizeof(OtterTensor*));
            for (int j = 0; j < network->num_start_of_line; j++) {
                input_arr[j] = inputs->dataset[idx][j];
            }
            for (int j = 0; j < network->num_end_of_line; j++) {
                label_arr[j] = labels->dataset[idx][j];
            }

            ON_SGD(network, input_arr, label_arr);

            for (int l = 0; l < network->num_layers; l++) {
                for (int k = 0; k < network->order[l]->weights_depth; k++) {
                    OT_ref_scalar_multiply(network->order[l]->weights_gradients[k], -network->learning_rate);
                    OT_ref_scalar_multiply(network->order[l]->biases_gradients[k], -network->learning_rate);
                    
                }
            }

            for (int l = 0; l < network->num_layers; l++) {
                for (int k = 0; k < network->order[l]->weights_depth; k++) {
                    OT_ref_tensors_sum(accum_weights[l][k], network->order[l]->weights_gradients[k], "ON_SGD_fit1");
                    OT_ref_tensors_sum(accum_biases[l][k], network->order[l]->biases_gradients[k], "ON_SGD_fit2");
                }
            }

            
            free(input_arr);
            free(label_arr);
        }
        for(int l=0;l<network->num_layers;l++){
            for(int k=0;k<network->order[l]->weights_depth;k++){
                OT_ref_copy(network->order[l]->weights_gradients[k], accum_weights[l][k]);
                OT_ref_copy(network->order[l]->biases_gradients[k], accum_biases[l][k]);
                OT_ref_scalar_multiply(network->order[l]->weights_gradients[k], inv_batch_size);
                OT_ref_scalar_multiply(network->order[l]->biases_gradients[k], inv_batch_size);
                OT_ref_reset(accum_weights[l][k]);
                OT_ref_reset(accum_biases[l][k]);

            }
        }
        
        ON_update_weights_and_biases(network);

            if (epoch %step==0 ||epoch==epochs) {ON_verbose1(epoch, network, inputs, labels, indices,batch_size);}
        free(indices);
    }
    free_clone_weights(network, accum_weights);
    free_clone_weights(network, accum_biases);
    free(accumulators);
    accumulators = NULL;
}



void ON_SGD(Otternetwork* network, OtterTensor** input, OtterTensor** labels) {
    int L = network->num_layers;

    OtterTensor** predictions = ON_feed_forward(network, input, 1);

    for (int i = 0; i < network->num_end_of_line; i++) {
        if (network->errors[i]) {
            free_malloc_tensor(&network->errors[i]);
        }
        network->errors[i] = ON_Cost_derivative(predictions[i], labels[i], network->error_function);
        if (!network->errors[i]) {
            fprintf(stderr, "Error: Null tensor encountered during cost derivative computation.\n");
            exit(EXIT_FAILURE);
        }
    }


    for (int i_layer = L - 1; i_layer >= 0; i_layer--) {
        switch (network->order[i_layer]->type) {
            case 0: // Dense layer
                ON_Dense_layer_backward(network, network->order[i_layer]);
                break;
            default:
                fprintf(stderr, "Unknown layer type for backward pass.\n");
                exit(EXIT_FAILURE);
        }
    }
    for(int i=0;i<network->num_end_of_line;i++){
        free_malloc_tensor(&predictions[i]);
    }
    free(predictions);
    predictions = NULL; 
    ON_reset_network(network);
}


void ON_verbose1(int epoch, Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int* indices,int batch_size) {
    printf("Epoch %d\n", epoch);
    for (int i = 0; i < network->num_end_of_line; i++) {
        printf("  Output %d:\n", i);
        float loss = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            int idx = indices ? indices[i] : i;
            OtterTensor** pred = ON_predict(network, inputs->dataset[idx]);
            loss += ON_cost(pred[i], labels->dataset[idx][i], network->error_function);
            free_ottertensor_list(pred, network->num_end_of_line);
            
        }
        printf("losses: %.3f\n", loss/batch_size);
    }

}





void ON_SGDM_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size) {
    OtterTensor**** accumulators = ON_init_clone_weights(network);
    OtterTensor*** accum_weights = accumulators[0];
    OtterTensor*** accum_biases  = accumulators[1];

    float inv_batch_size = 1.0f / (float)batch_size;
    float coef = -network->learning_rate*inv_batch_size;
    float momentum_coeff = network->optimizer_params[0]; 
    int step = epochs / 9;
    Momentums* momentums = init_first_Momentums(network);

    for (int epoch = 0; epoch < epochs; epoch++) {
        if(epoch % 500==0 && epoch!=0){
            network->learning_rate*=0.9f;
            coef = -network->learning_rate*inv_batch_size;
        }
        int* indices = OR_select_batch(inputs->size[0], batch_size);

        for (int i = 0; i < batch_size; i++) {
            int idx = indices[i];
            OtterTensor** input_arr = calloc(network->num_start_of_line, sizeof(OtterTensor*));
            OtterTensor** label_arr = calloc(network->num_end_of_line, sizeof(OtterTensor*));
            for (int j = 0; j < network->num_start_of_line; j++) {
                input_arr[j] = inputs->dataset[idx][j];
            }
            for (int j = 0; j < network->num_end_of_line; j++) {
                label_arr[j] = labels->dataset[idx][j];
            }

            ON_SGD(network, input_arr, label_arr);

            for (int l = 0; l < network->num_layers; l++) {
                for (int k = 0; k < network->order[l]->weights_depth; k++) {
                    OT_ref_tensors_sum(accum_weights[l][k], network->order[l]->weights_gradients[k], "ON_SGD_fit1");
                    OT_ref_tensors_sum(accum_biases[l][k], network->order[l]->biases_gradients[k], "ON_SGD_fit2");
                }
            }

            
            free(input_arr);
            input_arr = NULL;
            free(label_arr);
            label_arr = NULL;
        }
        for(int l=0;l<network->num_layers;l++){
            for(int k=0;k<network->order[l]->weights_depth;k++){
                OT_ref_scalar_multiply(accum_weights[l][k], coef);
                OT_ref_scalar_multiply(accum_biases[l][k], coef);
                    
                OT_ref_scalar_multiply(momentums[l].first_moment_weights[k], momentum_coeff);
                OT_ref_scalar_multiply(momentums[l].first_moment_biases[k], momentum_coeff);
                
                OT_ref_tensors_sum(momentums[l].first_moment_weights[k], accum_weights[l][k], "ON_SGDM_fit1");
                OT_ref_tensors_sum(momentums[l].first_moment_biases[k], accum_biases[l][k], "ON_SGDM_fit2");

                OT_ref_copy(network->order[l]->weights_gradients[k], momentums[l].first_moment_weights[k]);
                OT_ref_copy(network->order[l]->biases_gradients[k], momentums[l].first_moment_biases[k]);
                
                OT_ref_reset(accum_weights[l][k]);
                OT_ref_reset(accum_biases[l][k]);

            }
        }
        ON_update_weights_and_biases(network);
        if  (epoch %step==0 ||epoch==epochs)  {ON_verbose1(epoch, network, inputs, labels, indices,batch_size);}
        free(indices);
        indices = NULL;
    }
    free_first_momentums(network,momentums);
    free_clone_weights(network, accum_weights);
    free_clone_weights(network, accum_biases);
    free(accumulators);
    accumulators = NULL;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------

Momentums* init_first_Momentums(Otternetwork* network) {
    Momentums* momentums = malloc(network->num_layers * sizeof(Momentums));
    for (int l = 0; l < network->num_layers; l++) {
        momentums[l].first_moment_weights = calloc(network->order[l]->weights_depth, sizeof(OtterTensor*));
        momentums[l].first_moment_biases = calloc(network->order[l]->weights_depth, sizeof(OtterTensor*));
        for (int k = 0; k < network->order[l]->weights_depth; k++) {
            momentums[l].first_moment_weights[k] = OT_zeros(network->order[l]->weights[k]->dims, network->order[l]->weights[k]->rank);
            momentums[l].first_moment_biases[k] = OT_zeros(network->order[l]->biases[k]->dims, network->order[l]->biases[k]->rank);
        }
    }
    return momentums;
}

void init_second_Momentums(Otternetwork* network,Momentums* momentums) {
    for (int l = 0; l < network->num_layers; l++) {
        momentums[l].second_moment_weights = calloc(network->order[l]->weights_depth, sizeof(OtterTensor*));
        momentums[l].second_moment_biases = calloc(network->order[l]->weights_depth, sizeof(OtterTensor*));
        for (int k = 0; k < network->order[l]->weights_depth; k++) {
            momentums[l].second_moment_weights[k] = OT_zeros(network->order[l]->weights[k]->dims, network->order[l]->weights[k]->rank);
            momentums[l].second_moment_biases[k] = OT_zeros(network->order[l]->biases[k]->dims, network->order[l]->biases[k]->rank);
        }
    }

    return;
}

void free_first_momentums(Otternetwork* network,Momentums* momentums) {
    if (!momentums) return;
    for (int l = 0; l < network->num_layers; l++) {

        for (int k = 0; k < network->order[l]->weights_depth; k++) {
            free_malloc_tensor(&momentums[l].first_moment_weights[k]);
            free_malloc_tensor(&momentums[l].first_moment_biases[k]);
        }
        free(momentums[l].first_moment_weights);
        momentums[l].first_moment_weights = NULL;
        free(momentums[l].first_moment_biases);
        momentums[l].first_moment_biases = NULL;
        
    }
    free(momentums);
    momentums = NULL;
}

void free_all_momentums(Otternetwork* network,Momentums* momentums) {
    if (!momentums) return;
    for (int l = 0; l < network->num_layers; l++) {

        for (int k = 0; k < network->order[l]->weights_depth; k++) {
            free_malloc_tensor(&momentums[l].first_moment_weights[k]);
            free_malloc_tensor(&momentums[l].first_moment_biases[k]);
            free_malloc_tensor(&momentums[l].second_moment_weights[k]);
            free_malloc_tensor(&momentums[l].second_moment_biases[k]);
        }
        free(momentums[l].first_moment_weights);
        momentums[l].first_moment_weights = NULL;
        free(momentums[l].first_moment_biases);
        momentums[l].first_moment_biases = NULL;
        free(momentums[l].second_moment_weights);
        momentums[l].second_moment_weights = NULL;
        free(momentums[l].second_moment_biases);
        momentums[l].second_moment_biases = NULL;
        
    }
    free(momentums);
    momentums = NULL;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------


void ON_update_first_moment(OtterTensor* momentum, OtterTensor* gradient, float beta1) {
    // m_t = beta1 * m_t-1 + (1 - beta1) * grad_t
    OT_ref_scalar_multiply(momentum, beta1);
    OtterTensor* prod=OT_scalar_multiply(gradient, 1 - beta1);
    OT_ref_tensors_sum(momentum, prod, "ON_update_first_moment");
    free_malloc_tensor(&prod);


}

void ON_update_second_moment(OtterTensor* momentum, OtterTensor* gradient, float beta2) {
    OT_ref_scalar_multiply(momentum, beta2);
    OtterTensor* grad_squared = OT_copy(gradient);
    OT_ref_square(grad_squared);
    OT_ref_scalar_multiply(grad_squared, 1 - beta2);
    OT_ref_tensors_sum(momentum, grad_squared, "ON_update_second_moment");
    free_malloc_tensor(&grad_squared);



}


void ON_update_gradients(Otternetwork* net, OtterTensor* first_moment, OtterTensor* second_moment, float epsilon,OtterTensor** weights_grad,float norm1, float norm2) {
    OtterTensor* sqrt_weights = OT_scalar_multiply(second_moment, norm2);
    OT_ref_sqrt(sqrt_weights);
    OT_ref_scalar_sum(sqrt_weights, epsilon);
    OtterTensor* pregrad = OT_scalar_multiply(first_moment, norm1);
    OT_ref_dot_divide(pregrad, sqrt_weights);
    free_malloc_tensor(&sqrt_weights);
    OT_ref_scalar_multiply(pregrad, -net->learning_rate);
    free_malloc_tensor(weights_grad);
    *weights_grad = pregrad;
}


void ON_moments_estimation(Otternetwork* net, Momentums* momentums,float beta1, float beta2,float epsilon,int t) {
    float first_normalizer = 1.0f / (1.0f - OM_int_power(beta1, t));
    float second_normalizer = 1.0f / (1.0f - OM_int_power(beta2, t));
    for(int l=0;l<net->num_layers;l++){
        for(int k=0;k<net->order[l]->weights_depth;k++){
            ON_update_first_moment(momentums[l].first_moment_weights[k], net->order[l]->weights_gradients[k], beta1);
            ON_update_first_moment(momentums[l].first_moment_biases[k], net->order[l]->biases_gradients[k], beta1);
            ON_update_second_moment(momentums[l].second_moment_weights[k], net->order[l]->weights_gradients[k], beta2);
            ON_update_second_moment(momentums[l].second_moment_biases[k], net->order[l]->biases_gradients[k], beta2);

            ON_update_gradients( net, momentums[l].first_moment_weights[k], momentums[l].second_moment_weights[k], epsilon, &net->order[l]->weights_gradients[k], first_normalizer, second_normalizer);
            ON_update_gradients( net, momentums[l].first_moment_biases[k], momentums[l].second_moment_biases[k], epsilon, &net->order[l]->biases_gradients[k], first_normalizer, second_normalizer);

        }
    }
}



void ON_Adam_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size) {
    float beta1 = network->optimizer_params[0]; 
    float beta2 = network->optimizer_params[1];
    float epsilon = network->optimizer_params[2];
    int step = epochs / 9;
    Momentums* momentums = init_first_Momentums(network);
    init_second_Momentums(network,momentums);
    int t=0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        int* indices = OR_select_batch(inputs->size[0], batch_size);

        for (int i = 0; i < batch_size; i++) {
            int idx = indices[i];
            OtterTensor** input_arr = calloc(network->num_start_of_line, sizeof(OtterTensor*));
            OtterTensor** label_arr = calloc(network->num_end_of_line, sizeof(OtterTensor*));
            for (int j = 0; j < network->num_start_of_line; j++) {
                input_arr[j] = inputs->dataset[idx][j];
            }
            for (int j = 0; j < network->num_end_of_line; j++) {
                label_arr[j] = labels->dataset[idx][j];
            }

            ON_SGD(network, input_arr, label_arr);

            t+=1;
            ON_moments_estimation(network, momentums, beta1, beta2,epsilon,t);


            ON_update_weights_and_biases(network);

            free(input_arr);
            input_arr = NULL;
            free(label_arr);
            label_arr = NULL;
        }
            if  (epoch %step==0 ||epoch==epochs)  {ON_verbose1(epoch, network, inputs, labels, indices,batch_size);}
        free(indices);
    }
    free_all_momentums(network,momentums);
}