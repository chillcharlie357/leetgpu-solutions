#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void solve(const float* A, const float* B, float* C, int N);

static void checkCuda(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
}

int main() {
    const int N = 1 << 20;

    std::vector<float> hA(N), hB(N), hC(N), hRef(N);
    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i) * 0.5f;
        hB[i] = static_cast<float>(i) * -0.25f;
        hRef[i] = hA[i] + hB[i];
    }

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    checkCuda(cudaMalloc(&dA, static_cast<size_t>(N) * sizeof(float)), "cudaMalloc dA");
    checkCuda(cudaMalloc(&dB, static_cast<size_t>(N) * sizeof(float)), "cudaMalloc dB");
    checkCuda(cudaMalloc(&dC, static_cast<size_t>(N) * sizeof(float)), "cudaMalloc dC");

    checkCuda(cudaMemcpy(dA, hA.data(), static_cast<size_t>(N) * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy hA->dA");
    checkCuda(cudaMemcpy(dB, hB.data(), static_cast<size_t>(N) * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy hB->dB");

    solve(dA, dB, dC, N);
    checkCuda(cudaGetLastError(), "kernel");

    checkCuda(cudaMemcpy(hC.data(), dC, static_cast<size_t>(N) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy dC->hC");

    float max_abs_err = 0.0f;
    int bad_i = -1;
    for (int i = 0; i < N; ++i) {
        float err = std::fabs(hC[i] - hRef[i]);
        if (err > max_abs_err) {
            max_abs_err = err;
            bad_i = i;
        }
    }

    if (max_abs_err > 1e-6f) {
        std::fprintf(stderr, "FAIL max_abs_err=%g at i=%d (got=%g ref=%g)\n", max_abs_err, bad_i, hC[bad_i], hRef[bad_i]);
        return 1;
    }

    std::printf("OK max_abs_err=%g\n", max_abs_err);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}

