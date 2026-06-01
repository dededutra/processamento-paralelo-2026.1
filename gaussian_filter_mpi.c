#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define RADIUS 2
#define KSIZE 5

// ==========================================================
// FUNÇÕES AUXILIARES
// ==========================================================

static double obter_tempo() {
    return MPI_Wtime();
}

static const char* nome_modo_balanceamento(int modo) {
    switch (modo) {
        case 1: return "Balanceado";
        case 2: return "Desbalanceamento leve";
        case 3: return "Desbalanceamento forte";
        case 4: return "Personalizado";
        default: return "Desconhecido";
    }
}

// Calcula rows_per_rank a partir de pesos inteiros.
// A distribuição garante ao menos RADIUS linhas por processo.
static void calcular_distribuicao_por_pesos(
    const int* pesos,
    int size,
    int H,
    int* rows_per_rank
) {
    int min_rows = RADIUS;
    int total_min = min_rows * size;

    if (H < total_min) {
        fprintf(stderr,
                "Altura insuficiente para a distribuicao escolhida.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int remaining = H - total_min;

    long long soma_pesos = 0;
    for (int i = 0; i < size; i++) {
        int p = pesos[i];
        if (p <= 0) p = 1;
        soma_pesos += p;
    }

    if (soma_pesos == 0) {
        soma_pesos = size;
    }

    int* extra = (int*) calloc(size, sizeof(int));
    double* frac = (double*) malloc(size * sizeof(double));

    if (!extra || !frac) {
        fprintf(stderr, "Erro de alocacao na distribuicao MPI.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int assigned = 0;
    for (int i = 0; i < size; i++) {
        int p = pesos[i];
        if (p <= 0) p = 1;

        double exato = (double) remaining * (double) p / (double) soma_pesos;
        extra[i] = (int) exato;
        frac[i] = exato - (double) extra[i];
        assigned += extra[i];
    }

    int sobrando = remaining - assigned;
    while (sobrando > 0) {
        int idx = 0;
        double melhor = frac[0];
        for (int i = 1; i < size; i++) {
            if (frac[i] > melhor) {
                melhor = frac[i];
                idx = i;
            }
        }
        extra[idx]++;
        frac[idx] = -1.0;
        sobrando--;
    }

    for (int i = 0; i < size; i++) {
        rows_per_rank[i] = min_rows + extra[i];
    }

    free(extra);
    free(frac);
}

static void preencher_distribuicao_balanceamento(
    int modo,
    int size,
    int H,
    int* rows_per_rank,
    int* pesos
) {
    if (modo == 1) {
        for (int i = 0; i < size; i++) {
            pesos[i] = 1;
        }
    } else if (modo == 2) {
        // Leve desbalanceamento: rank 0 recebe mais linhas que os demais.
        for (int i = 0; i < size; i++) {
            pesos[i] = size - i;
        }
    } else if (modo == 3) {
        // Forte desbalanceamento: distribuição quadrática decrescente.
        for (int i = 0; i < size; i++) {
            int v = size - i;
            pesos[i] = v * v;
        }
    } else {
        // Personalizado: pesos já devem ter sido preenchidos pelo usuário.
    }

    calcular_distribuicao_por_pesos(pesos, size, H, rows_per_rank);
}

static void imprimir_distribuicao_mpi(
    int rank,
    int size,
    int modo_balanceamento,
    const int* rows_per_rank
) {
    if (rank != 0) return;

    printf("\n--- DISTRIBUICAO MPI ---\n");
    printf("Modo: %s\n", nome_modo_balanceamento(modo_balanceamento));
    printf("--------------------------------------------------\n");
    printf("Rank | Linhas atribuídas\n");
    printf("--------------------------------------------------\n");
    for (int i = 0; i < size; i++) {
        printf("%4d | %16d\n", i, rows_per_rank[i]);
    }
    printf("--------------------------------------------------\n");
}

// ==========================================================
// EXECUÇÃO MPI
// ==========================================================

static void executar_mpi(
    int* imagem_original,
    int* imagem_resultado_global,
    int W,
    int H,
    int num_iteracoes,
    const int kernel[KSIZE][KSIZE],
    int soma_pesos,
    int rank,
    int size,
    const int* rows_per_rank,
    double* tempo_comunicacao_local,
    double* tempo_computacao_local
) {
    int local_rows = rows_per_rank[rank];

    int global_start = 0;
    for (int r = 0; r < rank; r++) {
        global_start += rows_per_rank[r];
    }
    int global_end = global_start + local_rows - 1;

    int local_alloc_rows = local_rows + 2 * RADIUS;
    int* local_in = (int*) calloc((size_t)local_alloc_rows * (size_t)W, sizeof(int));
    int* local_out = (int*) calloc((size_t)local_alloc_rows * (size_t)W, sizeof(int));

    if (!local_in || !local_out) {
        fprintf(stderr, "Erro de alocacao local MPI no rank %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int* sendcounts = NULL;
    int* displs = NULL;

    if (rank == 0) {
        sendcounts = (int*) malloc(size * sizeof(int));
        displs = (int*) malloc(size * sizeof(int));
        if (!sendcounts || !displs) {
            fprintf(stderr, "Erro de alocacao de sendcounts/displs.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        displs[0] = 0;
        for (int r = 0; r < size; r++) {
            sendcounts[r] = rows_per_rank[r] * W;
            if (r > 0) {
                displs[r] = displs[r - 1] + sendcounts[r - 1];
            }
        }
    }

    // Distribui os blocos da imagem.
    MPI_Scatterv(
        imagem_original,
        sendcounts,
        displs,
        MPI_INT,
        local_in + RADIUS * W,
        local_rows * W,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Mantém o buffer de saída inicializado com a imagem original local.
    memcpy(
        local_out,
        local_in,
        (size_t)local_alloc_rows * (size_t)W * sizeof(int)
    );

    int* leitura = local_in;
    int* escrita = local_out;

    *tempo_comunicacao_local = 0.0;
    *tempo_computacao_local = 0.0;

    for (int iter = 0; iter < num_iteracoes; iter++) {
        double t_com_inicio = obter_tempo();

        // Troca de halos superiores.
        if (rank > 0) {
            MPI_Sendrecv(
                leitura + RADIUS * W,
                RADIUS * W,
                MPI_INT,
                rank - 1,
                100,
                leitura,
                RADIUS * W,
                MPI_INT,
                rank - 1,
                101,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        }

        // Troca de halos inferiores.
        if (rank < size - 1) {
            MPI_Sendrecv(
                leitura + local_rows * W,
                RADIUS * W,
                MPI_INT,
                rank + 1,
                101,
                leitura + (local_rows + RADIUS) * W,
                RADIUS * W,
                MPI_INT,
                rank + 1,
                100,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        }

        double t_com_fim = obter_tempo();
        *tempo_comunicacao_local += (t_com_fim - t_com_inicio);

        double t_comp_inicio = obter_tempo();

        int y_ini = global_start;
        if (y_ini < RADIUS) {
            y_ini = RADIUS;
        }

        int y_fim = global_end;
        if (y_fim > H - RADIUS - 1) {
            y_fim = H - RADIUS - 1;
        }

        for (int y = y_ini; y <= y_fim; y++) {
            int ly = (y - global_start) + RADIUS;
            for (int x = RADIUS; x < W - RADIUS; x++) {
                int sum = 0;
                for (int j = -RADIUS; j <= RADIUS; j++) {
                    for (int k = -RADIUS; k <= RADIUS; k++) {
                        int peso = kernel[j + RADIUS][k + RADIUS];
                        sum += leitura[(ly + j) * W + (x + k)] * peso;
                    }
                }
                escrita[ly * W + x] = sum / soma_pesos;
            }
        }

        double t_comp_fim = obter_tempo();
        *tempo_computacao_local += (t_comp_fim - t_comp_inicio);

        int* temp = leitura;
        leitura = escrita;
        escrita = temp;
    }

    // Coleta o resultado final no processo raiz.
    MPI_Gatherv(
        leitura + RADIUS * W,
        local_rows * W,
        MPI_INT,
        imagem_resultado_global,
        sendcounts,
        displs,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    free(local_in);
    free(local_out);

    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }
}

// ==========================================================
// MAIN
// ==========================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int W;
    int H;
    int num_iteracoes;
    int modo_balanceamento = 1;

    if (rank == 0) {
        printf("Tamanho da Imagem: WxH\n");
        scanf("%d %*[xX ] %d", &W, &H);
        //scanf("%d", &W);
        //scanf("%d", &H);

        printf("Numero de Iteracoes (default 10): ");
        if (scanf("%d", &num_iteracoes) != 1) {
            num_iteracoes = 10;
        }

        printf("\nModo de balanceamento MPI:\n");
        printf("1 - Balanceado\n");
        printf("2 - Desbalanceamento leve\n");
        printf("3 - Desbalanceamento forte\n");
        printf("4 - Personalizado (pesos por processo)\n");
        printf("Opcao: ");
        if (scanf("%d", &modo_balanceamento) != 1) {
            modo_balanceamento = 1;
        }

        if (modo_balanceamento < 1 || modo_balanceamento > 4) {
            modo_balanceamento = 1;
        }
    }

    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_iteracoes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&modo_balanceamento, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (W < 5 || H < 5) {
        if (rank == 0) {
            printf("A imagem precisa ter ao menos 5x5 para o kernel 5x5.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Garantia para manter pelo menos 2 linhas por processo.
    if (H < 2 * size) {
        if (rank == 0) {
            printf("O numero de processos MPI deve ser menor ou igual a H/2 para este stencil.\n");
        }
        MPI_Finalize();
        return 0;
    }

    int kernel[KSIZE][KSIZE] = {
        {1,  4,  6,  4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1,  4,  6,  4, 1}
    };
    int soma_pesos = 256;

    size_t tamanho_memoria = (size_t) W * (size_t) H * sizeof(int);

    int* imagem_original = NULL;
    int* img_seq = NULL;
    int* img_temp_seq = NULL;
    int* img_mpi = NULL;

    if (rank == 0) {
        imagem_original = (int*) malloc(tamanho_memoria);
        img_seq = (int*) malloc(tamanho_memoria);
        img_temp_seq = (int*) malloc(tamanho_memoria);
        img_mpi = (int*) malloc(tamanho_memoria);

        if (!imagem_original || !img_seq || !img_temp_seq || !img_mpi) {
            fprintf(stderr, "Erro de alocacao de memoria no processo raiz.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        srand(12345);
        for (int i = 0; i < W * H; i++) {
            imagem_original[i] = rand() % 256;
        }

        memcpy(img_seq, imagem_original, tamanho_memoria);
        memcpy(img_temp_seq, imagem_original, tamanho_memoria);
        memcpy(img_mpi, imagem_original, tamanho_memoria);
    }

    // Distribuição MPI definida via menu.
    int* rows_per_rank = (int*) malloc(size * sizeof(int));
    if (!rows_per_rank) {
        fprintf(stderr, "Erro de alocacao de rows_per_rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        int* pesos = (int*) malloc(size * sizeof(int));
        if (!pesos) {
            fprintf(stderr, "Erro de alocacao de pesos MPI.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (modo_balanceamento == 4) {
            printf("\nDigite os pesos de cada processo (inteiros positivos).\n");
            for (int i = 0; i < size; i++) {
                printf("Peso do rank %d: ", i);
                if (scanf("%d", &pesos[i]) != 1 || pesos[i] <= 0) {
                    pesos[i] = 1;
                }
            }
        } else {
            preencher_distribuicao_balanceamento(modo_balanceamento, size, H, rows_per_rank, pesos);
        }

        if (modo_balanceamento == 4) {
            calcular_distribuicao_por_pesos(pesos, size, H, rows_per_rank);
        }

        free(pesos);
    }

    MPI_Bcast(rows_per_rank, size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        imprimir_distribuicao_mpi(rank, size, modo_balanceamento, rows_per_rank);
    }

    // ======================================================
    // EXECUÇÃO SEQUENCIAL (Apenas para base de correção)
    // ======================================================

    double tempo_seq = 0.0;
    if (rank == 0) {
        printf("Rodando versao Sequencial para baseline...\n");
        int* ptr_leitura = img_seq;
        int* ptr_escrita = img_temp_seq;

        double inicio_seq = obter_tempo();

        for (int iter = 0; iter < num_iteracoes; iter++) {
            for (int y = RADIUS; y < H - RADIUS; y++) {
                for (int x = RADIUS; x < W - RADIUS; x++) {
                    int sum = 0;
                    for (int j = -RADIUS; j <= RADIUS; j++) {
                        for (int k = -RADIUS; k <= RADIUS; k++) {
                            int peso = kernel[j + RADIUS][k + RADIUS];
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
        tempo_seq = fim_seq - inicio_seq;
    }

    // ======================================================
    // EXECUÇÃO MPI
    // ======================================================

    MPI_Barrier(MPI_COMM_WORLD);
    double inicio_mpi_local = obter_tempo();

    double tempo_comm_local = 0.0;
    double tempo_comp_local = 0.0;

    executar_mpi(
        imagem_original,
        img_mpi,
        W,
        H,
        num_iteracoes,
        kernel,
        soma_pesos,
        rank,
        size,
        rows_per_rank,
        &tempo_comm_local,
        &tempo_comp_local
    );

    double fim_mpi_local = obter_tempo();
    double tempo_mpi_local = fim_mpi_local - inicio_mpi_local;

    double tempo_mpi_total = 0.0;
    double tempo_mpi_comm = 0.0;
    double tempo_mpi_comp = 0.0;

    MPI_Reduce(&tempo_mpi_local, &tempo_mpi_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tempo_comm_local, &tempo_mpi_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tempo_comp_local, &tempo_mpi_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ======================================================
    // RESULTADOS
    // ======================================================

    if (rank == 0) {
        printf("\n");
        printf("=====================================================================================================================\n");
        printf("| IMPLEMENTACAO | WORKERS | TEMPO TOTAL (s) | TEMPO COMP (s) | TEMPO COM (s) | SPEEDUP | EFICIENCIA |\n");
        printf("=====================================================================================================================\n");

        printf("| Sequencial    | %7d | %16.6f | %15s | %14s | %8.2f | %10.2f%% |\n",
               1,
               tempo_seq,
               "-",
               "-",
               1.0,
               100.0);

        if (tempo_mpi_total > 0.0) {
            double speedup_mpi = tempo_seq / tempo_mpi_total;
            double eficiencia_mpi = (speedup_mpi / size) * 100.0;

            printf("| MPI           | %7d | %16.6f | %15.6f | %14.6f | %8.2f | %10.2f%% |\n",
                   size,
                   tempo_mpi_total,
                   tempo_mpi_comp,
                   tempo_mpi_comm,
                   speedup_mpi,
                   eficiencia_mpi);
        }

        printf("=====================================================================================================================\n");

        printf("\n--- VALIDACAO DE CORRETUDE ---\n");

        int* result_seq = (num_iteracoes % 2 == 0) ? img_seq : img_temp_seq;
        int erros_mpi = 0;

        for (int i = 0; i < W * H; i++) {
            if (result_seq[i] != img_mpi[i]) {
                erros_mpi++;
                if (erros_mpi <= 3) {
                    printf("Divergencia MPI no indice %d -> Seq: %d | MPI: %d\n",
                           i, result_seq[i], img_mpi[i]);
                }
            }
        }

        if (erros_mpi == 0) {
            printf("[SUCESSO] A matriz MPI e exatamente igual a matriz sequencial!\n");
        } else {
            printf("[FALHA] Foram encontrados %d pixels divergentes no MPI.\n", erros_mpi);
        }
    }

    // ======================================================
    // LIBERAÇÃO DE MEMÓRIA
    // ======================================================

    free(rows_per_rank);

    if (rank == 0) {
        free(imagem_original);
        free(img_seq);
        free(img_temp_seq);
        free(img_mpi);
    }

    MPI_Finalize();
    return 0;
}