#include "../header/Ottercuda.h"


int g_cuda_disponible = 0;


void OTC_init() {
    #ifdef USE_CUDA // Cette macro est définie à la compilation si CUDA est inclus
    
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err == cudaSuccess && deviceCount > 0) {
        g_cuda_disponible = true;
        printf("Otter: CUDA détecté. Utilisation du GPU activée.\n");
    } else {
        g_cuda_disponible = false;
        printf("Otter: Aucun GPU CUDA détecté. Utilisation du CPU (fallback).\n");
    }

    #else
    // Si le code a été compilé SANS CUDA, g_cuda_disponible reste 'false'
    printf("Otter: Compilé sans CUDA. Utilisation du CPU.\n");
    #endif
}


// Déplace un tenseur vers le GPU, si possible.
void OT_to_cuda(OtterTensor* t) {
    if (t->device == DEVICE_CUDA) {
        return; // Déjà sur le GPU
    }

    if (g_cuda_disponible) {
        // --- On peut VRAIMENT y aller ---
        #ifdef USE_CUDA
        cudaMalloc(&t->d_data, t->size * sizeof(float));
        cudaMemcpy(t->d_data, t->h_data, ..., cudaMemcpyHostToDevice);
        t->device = DEVICE_CUDA;
        free(t->h_data); // On libère la RAM
        t->h_data = NULL;
        #endif
    } else {
        // --- Pas de GPU ---
        printf("Avertissement: OT_to_cuda() appelé mais aucun GPU n'est disponible. Le tenseur reste sur CPU.\n");
        // Le tenseur reste t->device == DEVICE_CPU
    }
}