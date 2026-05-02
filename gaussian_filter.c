#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

double get_time() {
    #ifdef _OPENMP
        return omp_get_wtime();
    #else
        return (double) clock() / CLOCKS_PER_SEC;
    #endif
}

int main(int argc, char** argv) {

    int W = 1280;
    int H = 720;
    int num_iteracoes = 10;

    int* imagem = (int*) malloc(W * H * sizeof(int));
    int* imagem_temp = (int*) malloc(W * H * sizeof(int));

    int* ptr_leitura = imagem;
    int* ptr_escrita = imagem_temp;

    srand(12345);

    #pragma omp parallel for
    for (int i = 0; i < W * H; i++) {
        ptr_leitura[i] = rand() % 256;
    }

    double start_time = get_time();

    for (int i = 0; i < num_iteracoes; i++) {
        
        #pragma omp parallel for
        for (int y = 2; y < H - 2; y++) {
            for (int x = 2; x < W - 2; x++) {
                
                // Apply Gaussian filter
                int sum = 0;
                for (int j = -2; j <= 2; j++) {
                    for (int k = -2; k <= 2; k++) {
                        sum += ptr_leitura[(y + j) * W + (x + k)];
                    }
                }

                ptr_escrita[y * W + x] = sum / 25;
            }
        }

        // Swap pointers
        int* temp = ptr_leitura;
        ptr_leitura = ptr_escrita;
        ptr_escrita = temp;
    }

    double end_time = get_time();
    printf("Tempo de execução: %f segundos\n", end_time - start_time);

    return 0;
}