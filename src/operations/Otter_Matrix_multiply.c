#include "../../header/operations/Otter_Matrix_multiply.h" // Doit contenir la struct OtterTensor, OT_zeros, etc.



void OT_Matrix_multiply_cpu(OtterTensor* a, OtterTensor* b, OtterTensor* result) {
    int M = a->dims[0];
    int K = a->dims[1];
    int N = b->dims[1];

    // Utilise les pointeurs CPU
    float* a_data = a->data;
    float* b_data = b->data;
    float* res_data = result->data;

    int j, k;
    float sum;

    #ifdef _OPENMP
    #pragma omp parallel for private(j, k, sum)
    #endif
    for (int i = 0; i < M; i++) {
        int a_row_offset = i * K;
        for (j = 0; j < N; j++) {
            sum = 0.0f;
            for (k = 0; k < K; k++) {
                sum += a_data[a_row_offset + k] * b_data[k * N + j];
            }
            res_data[i * N + j] = sum;
        }
    }
}

// ===============================================
// == IMPLÉMENTATION 2: GPU (CUDA)
// ===============================================

void OT_Matrix_multiply_cuda(OtterTensor* a, OtterTensor* b, OtterTensor* result) {
    #ifdef USE_CUDA
    // S'assurer que les données d'entrée sont bien sur le GPU
    if (a->gpu_data == NULL || b->gpu_data == NULL) {
        fprintf(stderr, "Erreur CUDA: Les tenseurs d'entrée ne sont pas sur le GPU (appelez OT_to_cuda).\n");
        exit(EXIT_FAILURE);
    }

    int M = a->dims[0];
    int K = a->dims[1];
    int N = b->dims[1];
    
    // Allouer la mémoire pour le résultat SUR LE GPU
    // (OT_zeros a alloué result->data, mais on a besoin de result->gpu_data)
    if (result->gpu_data == NULL) {
        cudaError_t err = cudaMalloc(&result->gpu_data, result->size * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "Erreur cudaMalloc: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    // Lancer le kernel (défini dans un fichier .cu)
    launch_matmul_kernel(a->gpu_data, b->gpu_data, result->gpu_data, M, K, N);
    
    // Note: le résultat est maintenant sur le GPU. 
    // Pour l'utiliser sur le CPU, il faudra appeler OT_to_cpu(result) plus tard.

    #else
    // Sécurité: Ne devrait jamais arriver si la logique du dispatcheur est correcte
    fprintf(stderr, "Erreur critique: OT_Matrix_multiply_cuda appelé sans compilation CUDA.\n");
    exit(EXIT_FAILURE);
    #endif
}


OtterTensor* OT_Matrix_multiply(OtterTensor* a, OtterTensor* b) {
    
    // --- 1. Vérifications (Dimensions) ---
    if (a->rank > 2 || b->rank > 2) {
        fprintf(stderr, "La multiplication n'est possible que pour les rangs 1 et 2.\n");
        exit(EXIT_FAILURE);
    }
    // ... (autres checks, ex: rank 0) ...
    if (a->dims[1] != b->dims[0]) {
        fprintf(stderr, "Les dimensions internes ne correspondent pas (%d != %d).\n", a->dims[1], b->dims[0]);
        exit(EXIT_FAILURE);
    }

    // --- 2. Vérification (Appareil) ---
    if (a->device != b->device) {
        fprintf(stderr, "Erreur: Tenseurs sur des appareils différents (CPU vs GPU).\n");
        // Idéalement, il faudrait déplacer 'b' sur 'a->device'
        exit(EXIT_FAILURE);
    }

    // --- 3. Allocation du tenseur résultat ---
    int M = a->dims[0];
    int N = b->dims[1];
    
    // OT_zeros alloue la structure et le pointeur CPU 'data'
    OtterTensor* result = OT_zeros((int[2]){M, N}, 2);
    // Le résultat vit sur le même appareil que les entrées
    result->device = a->device;

    // --- 4. Aiguillage (Dispatch) ---
    if (a->device == DEVICE_CUDA) {
        
        #ifdef USE_CUDA
        // Appelle la version GPU
        OT_Matrix_multiply_cuda(a, b, result);
        #else
        // Erreur si le tenseur est sur GPU mais le code n'a pas été compilé avec CUDA
        fprintf(stderr, "Erreur: Le tenseur est sur GPU, mais le code n'est pas compilé avec USE_CUDA.\n");
        exit(EXIT_FAILURE);
        #endif

    } else { // (a->device == DEVICE_CPU)
        
        // Appelle la version CPU
        OT_Matrix_multiply_cpu(a, b, result);
    }

    return result;
}