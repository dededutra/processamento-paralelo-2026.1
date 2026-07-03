#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define _CRT_SECURE_NO_WARNINGS

#define RADIUS 2
#define KSIZE 5
#define EPSILON 1e-4f
#define BLOCK_DIM 16

#define CUDA_CHECK(comando)                                                                    \
    do                                                                                         \
    {                                                                                          \
        cudaError_t err = (comando);                                                           \
        if (err != cudaSuccess)                                                                \
        {                                                                                      \
            printf("\n[ERRO FATAL CUDA] %s na linha %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

// Memória constante inicializada em tempo de compilação na GPU
__constant__ float d_kernel[KSIZE][KSIZE] = {
    {1.0f / 256.0f, 4.0f / 256.0f, 6.0f / 256.0f, 4.0f / 256.0f, 1.0f / 256.0f},
    {4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f},
    {6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f},
    {4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f},
    {1.0f / 256.0f, 4.0f / 256.0f, 6.0f / 256.0f, 4.0f / 256.0f, 1.0f / 256.0f}};

const float h_kernel[KSIZE][KSIZE] = {
    {1.0f / 256.0f, 4.0f / 256.0f, 6.0f / 256.0f, 4.0f / 256.0f, 1.0f / 256.0f},
    {4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f},
    {6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f},
    {4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f},
    {1.0f / 256.0f, 4.0f / 256.0f, 6.0f / 256.0f, 4.0f / 256.0f, 1.0f / 256.0f}};

double obter_tempo()
{
    auto agora = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tempo_em_segundos = agora.time_since_epoch();
    return tempo_em_segundos.count();
}

__global__ void filtro_gaussiano_kernel(const float *__restrict__ leitura, float *__restrict__ escrita, int W, int H)
{
    __shared__ float s_img[BLOCK_DIM + 2 * RADIUS][BLOCK_DIM + 2 * RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int tid = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y;

    int s_W = blockDim.x + 2 * RADIUS;
    int s_H = blockDim.y + 2 * RADIUS;
    int total_shared_elements = s_W * s_H;

    int x_offset = blockIdx.x * blockDim.x - RADIUS;
    int y_offset = blockIdx.y * blockDim.y - RADIUS;

    for (int i = tid; i < total_shared_elements; i += total_threads)
    {
        int s_ty = i / s_W;
        int s_tx = i % s_W;
        int g_x = x_offset + s_tx;
        int g_y = y_offset + s_ty;

        if (g_x >= 0 && g_x < W && g_y >= 0 && g_y < H)
        {
            s_img[s_ty][s_tx] = leitura[g_y * W + g_x];
        }
        else
        {
            s_img[s_ty][s_tx] = 0.0f;
        }
    }

    __syncthreads();

    if (x >= RADIUS && x < W - RADIUS && y >= RADIUS && y < H - RADIUS)
    {
        float sum = 0.0f;
        int s_cx = tx + RADIUS;
        int s_cy = ty + RADIUS;

        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            for (int k = -RADIUS; k <= RADIUS; k++)
            {
                float peso = d_kernel[j + RADIUS][k + RADIUS];
                sum += s_img[s_cy + j][s_cx + k] * peso;
            }
        }
        escrita[y * W + x] = sum;
    }
}

int main(int argc, char **argv)
{
    int W, H, num_iteracoes;

    printf("Tamanho da Imagem: W H\n");
    if (scanf("%d %d", &W, &H) != 2)
        return -1;

    printf("Numero de Iteracoes (default 10): ");
    if (scanf("%d", &num_iteracoes) != 1)
        num_iteracoes = 10;

    if (W < 5 || H < 5)
        return 0;

    size_t tamanho_memoria = (size_t)W * (size_t)H * sizeof(float);

    float *imagem_original = (float *)malloc(tamanho_memoria);
    float *img_seq = (float *)malloc(tamanho_memoria);
    float *img_temp_seq = (float *)malloc(tamanho_memoria);
    float *img_cuda = (float *)malloc(tamanho_memoria);

    srand(12345);
    for (int i = 0; i < W * H; i++)
        imagem_original[i] = (float)(rand() % 256);

    memcpy(img_seq, imagem_original, tamanho_memoria);
    memcpy(img_temp_seq, imagem_original, tamanho_memoria);

    float *ptr_leitura_seq = img_seq;
    float *ptr_escrita_seq = img_temp_seq;
    double inicio_seq = obter_tempo();

    for (int iter = 0; iter < num_iteracoes; iter++)
    {
        for (int y = RADIUS; y < H - RADIUS; y++)
        {
            for (int x = RADIUS; x < W - RADIUS; x++)
            {
                float sum = 0.0f;
                for (int j = -RADIUS; j <= RADIUS; j++)
                {
                    for (int k = -RADIUS; k <= RADIUS; k++)
                    {
                        sum += ptr_leitura_seq[(y + j) * W + (x + k)] * h_kernel[j + RADIUS][k + RADIUS];
                    }
                }
                ptr_escrita_seq[y * W + x] = sum;
            }
        }
        float *temp = ptr_leitura_seq;
        ptr_leitura_seq = ptr_escrita_seq;
        ptr_escrita_seq = temp;
    }
    double tempo_seq = obter_tempo() - inicio_seq;

    printf("\nRodando versao NVIDIA CUDA (com Shared Memory)...\n");

    float *d_leitura, *d_escrita;

    CUDA_CHECK(cudaMalloc((void **)&d_leitura, tamanho_memoria));
    CUDA_CHECK(cudaMalloc((void **)&d_escrita, tamanho_memoria));

    double inicio_comm = obter_tempo();

    CUDA_CHECK(cudaMemcpy(d_leitura, imagem_original, tamanho_memoria, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_escrita, imagem_original, tamanho_memoria, cudaMemcpyHostToDevice));
    double tempo_comunicacao = obter_tempo() - inicio_comm;

    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 numBlocks((W + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (H + threadsPerBlock.y - 1) / threadsPerBlock.y);

    CUDA_CHECK(cudaDeviceSynchronize());
    double inicio_comp = obter_tempo();

    for (int iter = 0; iter < num_iteracoes; iter++)
    {

        filtro_gaussiano_kernel<<<numBlocks, threadsPerBlock>>>(d_leitura, d_escrita, W, H);
        CUDA_CHECK(cudaGetLastError());

        float *d_temp = d_leitura;
        d_leitura = d_escrita;
        d_escrita = d_temp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double tempo_computacao = obter_tempo() - inicio_comp;

    double inicio_download = obter_tempo();
    CUDA_CHECK(cudaMemcpy(img_cuda, d_leitura, tamanho_memoria, cudaMemcpyDeviceToHost));
    tempo_comunicacao += (obter_tempo() - inicio_download);

    double tempo_cuda_total = tempo_computacao + tempo_comunicacao;

    printf("===================================================================================================\n");
    printf("| IMPLEMENTACAO | TEMPO TOTAL (s) | TEMPO COMP (s) | TEMPO COM (s) | SPEEDUP GLOBAL |\n");
    printf("===================================================================================================\n");
    printf("| Sequencial    | %15.6f | %14.6f | %13s | %14.2fx |\n", tempo_seq, tempo_seq, "-", 1.0);
    if (tempo_cuda_total > 0.0)
    {
        printf("| NVIDIA CUDA   | %15.6f | %14.6f | %14.6f | %14.2fx |\n",
               tempo_cuda_total, tempo_computacao, tempo_comunicacao, tempo_seq / tempo_cuda_total);
        printf("---------------------------------------------------------------------------------------------------\n");
        printf("(*) Speedup isolado do Kernel CUDA Otimizado: %.2fx\n", tempo_seq / tempo_computacao);
    }
    printf("===================================================================================================\n");

    int erros_cuda = 0;
    float *resultado_final_seq = (num_iteracoes % 2 == 0) ? img_seq : img_temp_seq;

    for (int i = 0; i < W * H; i++)
    {
        float dif = fabs(resultado_final_seq[i] - img_cuda[i]);
        if (dif > EPSILON)
        {
            erros_cuda++;
            if (erros_cuda <= 3)
            {
                printf("Divergencia CUDA no indice %d -> Seq: %.6f | CUDA: %.6f | Dif: %.6f\n",
                       i, resultado_final_seq[i], img_cuda[i], dif);
            }
        }
    }

    if (erros_cuda == 0)
        printf("\n[SUCESSO] A matriz CUDA otimizada e equivalente a matriz sequencial!\n");
    else
        printf("\n[FALHA] Foram encontrados %d pixels divergentes fora da tolerancia.\n", erros_cuda);

    cudaFree(d_leitura);
    cudaFree(d_escrita);
    free(imagem_original);
    free(img_seq);
    free(img_temp_seq);
    free(img_cuda);

    return 0;
}