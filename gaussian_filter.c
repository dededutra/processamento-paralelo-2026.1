#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

// Função para medir tempo (funciona com ou sem a flag -fopenmp)
double obter_tempo() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

// Função para obter o número de threads
int obter_num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1; // Sem OpenMP, apenas 1 thread
#endif
}

int main(int argc, char** argv) {

    int W;
    int H;
    int num_iteracoes;

    printf("Tamanho da Imagem: WxL\n");
    scanf("%d", &W);
    scanf("%d", &H);

    printf("Numero de Iteracoes (default 10): ");
    scanf("%d", &num_iteracoes) == 1 ? num_iteracoes : 10; // Permite passar o número de iterações como argumento

    // Definição do Kernel Gaussiano 5x5 e soma dos pesos
    int kernel[5][5] = {
        {1,  4,  6,  4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1,  4,  6,  4, 1}
    };
    int soma_pesos = 256;

    // 1. Alocação de Memória
    size_t tamanho_memoria = W * H * sizeof(int);
    
    int* imagem_original = (int*) malloc(tamanho_memoria);
    
    // Arrays para o teste Sequencial
    int* img_seq      = (int*) malloc(tamanho_memoria);
    int* img_temp_seq = (int*) malloc(tamanho_memoria);
    
    // Arrays para o teste Paralelo
    int* img_par      = (int*) malloc(tamanho_memoria);
    int* img_temp_par = (int*) malloc(tamanho_memoria);

    // 2. Inicialização da Imagem Original com "ruído"
    srand(12345);
    for (int i = 0; i < W * H; i++) {
        imagem_original[i] = rand() % 256;
    }

    // Copiamos a original para os arrays de trabalho (isso resolve o problema das bordas!)
    memcpy(img_seq, imagem_original, tamanho_memoria);
    memcpy(img_temp_seq, imagem_original, tamanho_memoria);
    memcpy(img_par, imagem_original, tamanho_memoria);
    memcpy(img_temp_par, imagem_original, tamanho_memoria);

    // ==========================================
    // EXECUÇÃO SEQUENCIAL (A Verdade Absoluta)
    // ==========================================
    printf("Rodando versao Sequencial...\n");
    int* ptr_leitura = img_seq;
    int* ptr_escrita = img_temp_seq;

    double inicio_seq = obter_tempo();

    for (int iter = 0; iter < num_iteracoes; iter++) {
        for (int y = 2; y < H - 2; y++) {
            for (int x = 2; x < W - 2; x++) {
                int sum = 0;
                for (int j = -2; j <= 2; j++) {
                    for (int k = -2; k <= 2; k++) {
                        int peso = kernel[j + 2][k + 2];
                        sum += ptr_leitura[(y + j) * W + (x + k)] * peso;
                    }
                }
                ptr_escrita[y * W + x] = sum / soma_pesos;
            }
        }
        int* temp = ptr_leitura;
        ptr_leitura = ptr_escrita;
        ptr_escrita = temp;
    }
    double fim_seq = obter_tempo();
    double tempo_seq = fim_seq - inicio_seq;

    // ==========================================
    // EXECUÇÃO PARALELA (Com OpenMP)
    // ==========================================
    printf("Rodando versao Paralela...\n");
    ptr_leitura = img_par; // Resetamos os ponteiros para os arrays paralelos
    ptr_escrita = img_temp_par;

    double inicio_par = obter_tempo();

    for (int iter = 0; iter < num_iteracoes; iter++) {
        
        #pragma omp parallel for
        for (int y = 2; y < H - 2; y++) {
            for (int x = 2; x < W - 2; x++) {
                int sum = 0;
                for (int j = -2; j <= 2; j++) {
                    for (int k = -2; k <= 2; k++) {
                        int peso = kernel[j + 2][k + 2];
                        sum += ptr_leitura[(y + j) * W + (x + k)] * peso;
                    }
                }
                ptr_escrita[y * W + x] = sum / soma_pesos;
            }
        }
        int* temp = ptr_leitura;
        ptr_leitura = ptr_escrita;
        ptr_escrita = temp;
    }
    double fim_par = obter_tempo();
    double tempo_par = fim_par - inicio_par;

    // ==========================================
    // RESULTADOS E TESTE UNITÁRIO
    // ==========================================
    printf("\n--- RESULTADOS DO BENCHMARK ---\n");
    printf("Tempo Sequencial: %f segundos\n", tempo_seq);
    printf("Tempo Paralelo:   %f segundos\n", tempo_par);
    
    if (tempo_par > 0) {
        double speedup = tempo_seq / tempo_par;
        int num_threads = obter_num_threads();
        double eficiencia = speedup / num_threads;

        printf("Speedup Real:     %.2fx\n", speedup);
        printf("Num. de Threads:  %d\n", num_threads);
        printf("Eficiencia:       %.2f%% (%.2f)\n", eficiencia * 100.0, eficiencia);
    }

    printf("\n--- VALIDACAO DE CORRETUDE ---\n");
    int erros = 0;
    
    // Como iteramos um número par de vezes (10), o resultado final está nos ponteiros base
    for (int i = 0; i < W * H; i++) {
        if (img_seq[i] != img_par[i]) {
            erros++;
            if (erros <= 3) { // Imprime apenas os 3 primeiros erros para não poluir a tela
                printf("Divergencia encontrada no indice %d -> Seq: %d | Par: %d\n", i, img_seq[i], img_par[i]);
            }
        }
    }

    if (erros == 0) {
        printf("[SUCESSO] A matriz paralela e exatemente igual a matriz sequencial!\n");
    } else {
        printf("[FALHA] Foram encontrados %d pixels divergentes.\n", erros);
    }

    // Liberação de Memória
    free(imagem_original);
    free(img_seq); free(img_temp_seq);
    free(img_par); free(img_temp_par);

    return 0;
}