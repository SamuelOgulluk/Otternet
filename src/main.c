#include "../header/main.h"





void display_order(Otternetwork* net) {
    printf("Network order:\n");
    for (int i = 0; i < net->num_layers; i++) {
        printf("layer : %p",(net->order[i]));
    }
    printf("\n");
}

void print_tensor_list(OtterTensor** tensors, int num_tensors, int significant_digits) {
    for (int i = 0; i < num_tensors; i++) {
        printf("Tensor %d:\n", i);
        print_tensor(tensors[i], significant_digits);
    }
}
OtterTensor** create_data(){
    int dims[2] = {4, 1}; // 4 features, 1 colonne
    OtterTensor* fake_input = OT_zeros(dims, 2);
    for (int i = 0; i < fake_input->size; i++) {
        fake_input->data[i] = (float)(i + 1); // [1, 2, 3, 4]
    }
    printf("Input tensor avant feed forward :\n");
    print_tensor(fake_input, 3);

    OtterTensor** input_array = malloc(sizeof(OtterTensor*));
    if (input_array == NULL) {
        fprintf(stderr, "Failed to allocate memory for input_array\n");
        free_malloc_tensor(&fake_input);
        exit(EXIT_FAILURE);
    }
    input_array[0] = fake_input;
    return input_array;
}


OtterDataset** create_data2(int full_size){


    int input_dims[2] = {4, 1};
    int target_dims0[2] = {1, 1}; 
    int target_dims1[2] = {2, 1}; 


    OtterTensor** fake_inputs_list = calloc(full_size,sizeof(OtterTensor*));
    if (fake_inputs_list == NULL) {
        fprintf(stderr, "Failed to allocate memory for fake_inputs_list\n");
        exit(EXIT_FAILURE);
    }
    OtterTensor** fake_targets_list0 = calloc(full_size,sizeof(OtterTensor*));
    if (fake_targets_list0 == NULL) {
        fprintf(stderr, "Failed to allocate memory for fake_targets_list0\n");
        free(fake_inputs_list);
        exit(EXIT_FAILURE);
    }
    OtterTensor** fake_targets_list1 = calloc(full_size,sizeof(OtterTensor*));
    if (fake_targets_list1 == NULL) {
        fprintf(stderr, "Failed to allocate memory for fake_targets_list1\n");
        free(fake_inputs_list);
        free(fake_targets_list0);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < full_size; i++) {
        fake_inputs_list[i] = OT_random_uniform(input_dims, 2,-1.0f, 1.0f);
        fake_targets_list0[i] = OT_random_uniform(target_dims0, 2,-1.0f, 1.0f);
        fake_targets_list1[i] = OT_random_uniform(target_dims1, 2,-1.0f, 1.0f);

    }



    OtterTensor*** labels = malloc(full_size * sizeof(OtterTensor**));
    if (labels == NULL) {
        fprintf(stderr, "Failed to allocate memory for labels\n");
        // Free previously allocated memory
        free(fake_inputs_list);
        free(fake_targets_list0);
        free(fake_targets_list1);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < full_size; i++) {
        labels[i] = malloc(2 * sizeof(OtterTensor*));
        if (labels[i] == NULL) {
            // Handle memory allocation failure
            // Free previously allocated memory
            for (int j = 0; j < i; j++) {
                free(labels[j]);
            }
            free(labels);
            free(fake_inputs_list);
            free(fake_targets_list0);
            free(fake_targets_list1);
            exit(EXIT_FAILURE);
        }
        labels[i][0] = fake_targets_list0[i];
        labels[i][1] = fake_targets_list1[i];
    }

    // forme : labels(full_size, 2, ...)

    OtterTensor*** inputs = malloc(full_size * sizeof(OtterTensor**));
    if (inputs == NULL) {
        fprintf(stderr, "Failed to allocate memory for inputs\n");
        // Free previously allocated memory
        for (int i = 0; i < full_size; i++) {
            free(labels[i]);
        }
        free(labels);
        free(fake_inputs_list);
        free(fake_targets_list0);
        free(fake_targets_list1);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < full_size; i++) {
        inputs[i] = malloc(1 * sizeof(OtterTensor*));
        if (inputs[i] == NULL) {
            // Handle memory allocation failure
            // Free previously allocated memory
            for (int j = 0; j < i; j++) {
                free(inputs[j]);
            }
            free(inputs);
            for (int j = 0; j < full_size; j++) {
                free(labels[j]);
            }
            free(labels);
            free(fake_inputs_list);
            free(fake_targets_list0);
            free(fake_targets_list1);
            exit(EXIT_FAILURE);
        }
        inputs[i][0] = fake_inputs_list[i];
    }

    //forme : inputs(full_size, 1, ...)


    printf("Input tensor avant fit :\n");
    print_tensor_list(inputs[2], 1, 3);



    OtterDataset* fake_inputs = Init_dataset(inputs,1,full_size);


    OtterDataset* fake_targets = Init_dataset(labels,2,full_size);


    OtterDataset** datasets = malloc(2 * sizeof(OtterDataset*));
    if (datasets == NULL) {
        fprintf(stderr, "Failed to allocate memory for datasets\n");
        OD_free_dataset(fake_inputs);
        OD_free_dataset(fake_targets);
        exit(EXIT_FAILURE);
    }
    datasets[0] = fake_inputs;
    datasets[1] = fake_targets;

    free(fake_inputs_list);
    free(fake_targets_list0);
    free(fake_targets_list1);



    return datasets;
}


Otternetwork* create_net(){


    // Construire le réseau
    Otterchain* dense_1 = ON_Dense_layer(4, "relu", NULL, 0,0);
    Otterchain* dense_2 = ON_Dense_layer(1, "relu", dense_1, 1,0);
    Otterchain* dense_3 = ON_Dense_layer(1, "relu", dense_2, 1,0);
    Otterchain* dense_4 = ON_Dense_layer(2, "tanh", dense_2, 1,0);
    Otterchain* dense_5 = ON_Dense_layer(2, "tanh", dense_4, 1,0);
    

    Otternetwork* net = ON_initialise_otternetwork();

    ON_add_layer(net, dense_1);
    ON_add_layer(net, dense_2);
    ON_add_layer(net, dense_3);
    ON_add_layer(net, dense_4);
    ON_add_layer(net, dense_5);

    ON_compile_otternetwork(net, "SGD", "MSE", 0.0001f, NULL);

    //ON_display_network(net);
    ON_display_network_connections(net);
    return net;
}


void test_ff(Otternetwork* net){
    OtterTensor** input_array = create_data();
    // === FEED FORWARD ===
    OtterTensor** output = ON_feed_forward(net, input_array, 1);
    printf("Output tensor après feed forward :\n");
    print_tensor_list(output, net->num_end_of_line, 2);
    free_ottertensor_list(output, net->num_end_of_line);
    free_ottertensor_list(input_array, 1);
    
}


void test_bp(Otternetwork* net){
    
    int full_size = 100;
    int batch_size = 10;

    OtterDataset** loutre = create_data2(full_size);
    OtterDataset* fake_inputs = loutre[0];
    OtterDataset* fake_targets = loutre[1];
    free(loutre);
    loutre = NULL;

    print_parameters(net);

    printf("Fitting sur fausses données...\n");

    ON_fit(net, fake_inputs, fake_targets, 1000, batch_size); // 10 epochs, batch_size

    printf("Fitting terminé.\n");


    OD_free_dataset(fake_inputs);
    OD_free_dataset(fake_targets);

    

    free_otternetwork(net);

    printf("Network freed\n");
    return ;
}





OtterDataset** create_toy_dataset(int full_size) {
    int input_dims[2] = {2, 1};
    int target_dims[2] = {1, 1};

    OtterTensor*** inputs = malloc(full_size * sizeof(OtterTensor**));
    if (inputs == NULL) {
        fprintf(stderr, "Failed to allocate memory for inputs\n");
        exit(EXIT_FAILURE);
    }
    OtterTensor*** labels = malloc(full_size * sizeof(OtterTensor**));
    if (labels == NULL) {
        fprintf(stderr, "Failed to allocate memory for labels\n");
        free(inputs);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < full_size; i++) {
        inputs[i] = malloc(1 * sizeof(OtterTensor*));
        if (inputs[i] == NULL) {
            // Handle memory allocation failure
            // Free previously allocated memory
            for (int j = 0; j < i; j++) {
                free(inputs[j]);
            }
            free(inputs);
            free(labels);
            exit(EXIT_FAILURE);
        }
        labels[i] = malloc(1 * sizeof(OtterTensor*));
        if (labels[i] == NULL) {
            // Handle memory allocation failure
            // Free previously allocated memory
            for (int j = 0; j < i; j++) {
                free(labels[j]);
            }
            free(labels);
            free(inputs[i]);
            free(inputs);
            exit(EXIT_FAILURE);
        }

        OtterTensor* x = OT_zeros(input_dims, 2);
        x->data[0] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // [-1,1]
        x->data[1] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        OtterTensor* y = OT_zeros(target_dims, 2);
        y->data[0] = sinf(x->data[0] * 3.14) + cosf(x->data[1] * 3.12); // toujours borné [-2,2]

        inputs[i][0] = x;
        labels[i][0] = y;
    }

    OtterDataset* input_ds = Init_dataset(inputs, 1, full_size);
    OtterDataset* label_ds = Init_dataset(labels, 1, full_size);

    OtterDataset** datasets = malloc(2 * sizeof(OtterDataset*));
    if (datasets == NULL) {
        fprintf(stderr, "Failed to allocate memory for datasets\n");
        OD_free_dataset(input_ds);
        OD_free_dataset(label_ds);
        exit(EXIT_FAILURE);
    }
    datasets[0] = input_ds;
    datasets[1] = label_ds;
    return datasets;
}


Otternetwork* create_toy_net() {
    // Dense layers avec initialisation Xavier / He
    Otterchain* dense_1 = ON_Dense_layer(16, "relu", NULL, 0, 2);
    Otterchain* dense_2 = ON_Dense_layer(16, "relu", dense_1, 1, 0);
    Otterchain* dense_3 = ON_Dense_layer(1, "linear", dense_2, 1, 0);


    Otternetwork* net = ON_initialise_otternetwork();
    ON_add_layer(net, dense_1);
    ON_add_layer(net, dense_2);
    ON_add_layer(net, dense_3);
    //float params[3]={0.9,0.999,1e-8};
    ON_compile_otternetwork(net, "Adam", "MSE", 0.001f, NULL); // lr ajusté
    return net;
}


void test_toy_training() {
    Otternetwork* net = create_toy_net();
    OtterDataset** data = create_toy_dataset(250);
    OtterDataset* inputs = data[0];
    OtterDataset* labels = data[1];
    free(data);

    printf("Avant entraînement :\n");
    OtterTensor** pred0 = ON_feed_forward(net, inputs->dataset[0], 0);
    print_tensor(pred0[0], 3);
    printf("Target = %.3f\n", labels->dataset[0][0]->data[0]);
    free_ottertensor_list(pred0, 1);

    ON_fit(net, inputs, labels, 2000, 32); // epochs=200, batch=5

    printf("Après entraînement :\n");
    OtterTensor** pred1 = ON_feed_forward(net, inputs->dataset[0], 0);
    print_tensor(pred1[0], 3);
    printf("Target = %.3f\n", labels->dataset[0][0]->data[0]);
    free_ottertensor_list(pred1, 1);

    OD_free_dataset(inputs);
    OD_free_dataset(labels);
    free_otternetwork(net);
}



int main() {
    //Otternetwork* net = create_net();
    OTC_init();

    //test_ff(net);
    test_toy_training();
    //test_bp(net);
}