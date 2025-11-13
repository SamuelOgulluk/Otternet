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


void OT_to_cuda(OtterTensor* t) {
    #ifndef USE_CUDA
        fprintf(stderr, "Avertissement : OT_to_cuda appelé mais USE_CUDA non défini.\n");
        return;
    #else

    // 2. Vérification matérielle (via votre init) et état actuel
    if (!g_cuda_disponible) {
        fprintf(stderr, "Avertissement : Pas de GPU détecté.\n");
        return;
    }
    if (t->device == DEVICE_CUDA) {
        return; // Déjà sur le GPU
    }
    if (t->data == NULL) {
        fprintf(stderr, "Erreur : Tentative de déplacement d'un tenseur vide vers GPU.\n");
        exit(EXIT_FAILURE);
    }

    // 3. Allocation de la mémoire GPU (si pas déjà allouée)
    if (t->gpu_data == NULL) {
        cudaError_t err = cudaMalloc((void**)&t->gpu_data, t->size * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "Erreur cudaMalloc : %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    // 4. Transfert des données (Hôte -> Appareil)
    cudaError_t err = cudaMemcpy(t->gpu_data, t->data, t->size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erreur cudaMemcpy : %s\n", cudaGetErrorString(err));
        // On ne libère pas data ici car le transfert a échoué
        exit(EXIT_FAILURE);
    }

    // 5. Mise à jour de l'état et nettoyage RAM
    t->device = DEVICE_CUDA;
    
    // Optionnel : Libérer la RAM CPU pour économiser de la place
    // Si vous commentez ces deux lignes, vous gardez une copie "cache" sur CPU (mais qui deviendra périmée)
    free(t->data);
    t->data = NULL; 

    #endif
}

void OT_to_cpu(OtterTensor* t) {
    // 1. Si pas de CUDA compilé, le tenseur est forcément déjà CPU
    #ifndef USE_CUDA
        return; 
    #else

    // 2. Si déjà sur CPU, on ne fait rien
    if (t->device == DEVICE_CPU) {
        return;
    }

    // Sécurité : Si le tenseur est marqué CUDA mais n'a pas de données
    if (t->gpu_data == NULL) {
        fprintf(stderr, "Erreur critique : Tenseur marqué DEVICE_CUDA mais gpu_data est NULL.\n");
        exit(EXIT_FAILURE);
    }

    // 3. Allocation de la mémoire CPU (car elle a été free dans OT_to_cuda)
    if (t->data == NULL) {
        t->data = malloc(t->size * sizeof(float));
        if (t->data == NULL) {
            fprintf(stderr, "Erreur : Échec allocation mémoire CPU dans OT_to_cpu.\n");
            exit(EXIT_FAILURE);
        }
    }

    // 4. Transfert des données (Appareil -> Hôte)
    cudaError_t err = cudaMemcpy(t->data, t->gpu_data, t->size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erreur cudaMemcpy (Device -> Host) : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // 5. Mise à jour de l'état et nettoyage VRAM
    t->device = DEVICE_CPU;
    
    // On libère la mémoire GPU car les données sont maintenant en sécurité sur le CPU
    cudaFree(t->gpu_data);
    t->gpu_data = NULL;

    #endif
}