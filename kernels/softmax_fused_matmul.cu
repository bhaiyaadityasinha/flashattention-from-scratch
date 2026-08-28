#include "common.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Fused softmax(Q @ K^T) — first step toward FlashAttention.
//
// S = QK^T is never written to global memory. One block owns one query row
// (same mapping as the other softmax kernels). The Q row is cached in shared
// memory; K is streamed in THREADS x TILE_D panels with the same cooperative
// load pattern as the shared-memory GEMM in cuda-matmul. Each panel of scores
// is merged into a running (max, sum) pair with merge_pair.
//
// The row max is not known until every key has been seen, so a second pass
// recomputes the dots and writes P = softmax(S). That recompute is the
// FlashAttention trade (extra FLOPs, no HBM for S). Full FlashAttention then
// fuses P @ V so P is not materialized either.
//
// Layout: Q [M, D], K [N, D], P [M, N], row-major float32.

constexpr int THREADS = 256;
constexpr int TILE_D = 32;

__device__ __forceinline__ float dot_k_tile(const float* __restrict__ K,
                                            const float* q_row,
                                            float* k_tile,
                                            int n0,
                                            int tid,
                                            int N,
                                            int D) {
    const int col = n0 + tid;
    float score = 0.0f;
    const int d_tiles = (D + TILE_D - 1) / TILE_D;
    for (int t = 0; t < d_tiles; ++t) {
        const int d0 = t * TILE_D;
        // Cooperative coalesced load: consecutive threads take consecutive D
        // in a THREADS x TILE_D panel of K. Same idea as matmul_smem.
        for (int idx = tid; idx < THREADS * TILE_D; idx += THREADS) {
            const int r = idx / TILE_D;
            const int c = idx % TILE_D;
            const int n = n0 + r;
            const int d = d0 + c;
            k_tile[idx] = (n < N && d < D)
                              ? K[static_cast<long long>(n) * D + d]
                              : 0.0f;
        }
        __syncthreads();
        if (col < N) {
#pragma unroll
            for (int c = 0; c < TILE_D; ++c) {
                if (d0 + c < D) {
                    score += q_row[d0 + c] * k_tile[tid * TILE_D + c];
                }
            }
        }
        __syncthreads();
    }
    return score;
}

__global__ void fused_matmul_softmax_kernel(const float* __restrict__ Q,
                                            const float* __restrict__ K,
                                            float* __restrict__ P,
                                            int M,
                                            int N,
                                            int D) {
    extern __shared__ float smem[];
    // q_row is padded to an even float count so k_tile (and later SoftmaxPair*)
    // is 8-byte aligned — same class of bug as softmax_tiled.cu.
    const int q_stride = (D + 1) & ~1;
    float* q_row = smem;
    float* k_tile = smem + q_stride;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= M) {
        return;
    }

    const float* q_src = Q + static_cast<long long>(row) * D;
    for (int d = tid; d < D; d += THREADS) {
        q_row[d] = q_src[d];
    }
    __syncthreads();

    SoftmaxPair local = empty_pair();
    for (int n0 = 0; n0 < N; n0 += THREADS) {
        const int col = n0 + tid;
        const float score = dot_k_tile(K, q_row, k_tile, n0, tid, N, D);
        if (col < N) {
            local = merge_pair(local, {score, 1.0f});
        }
    }

    // Reuse the K-tile buffer for the block reduction (it is idle between passes).
    SoftmaxPair* partial = reinterpret_cast<SoftmaxPair*>(k_tile);
    partial[tid] = local;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] = merge_pair(partial[tid], partial[tid + stride]);
        }
        __syncthreads();
    }
    const SoftmaxPair total = partial[0];
    const float inv_d = 1.0f / total.d;

    float* row_out = P + static_cast<long long>(row) * N;
    for (int n0 = 0; n0 < N; n0 += THREADS) {
        const int col = n0 + tid;
        const float score = dot_k_tile(K, q_row, k_tile, n0, tid, N, D);
        if (col < N) {
            row_out[col] = expf(score - total.m) * inv_d;
        }
    }
}

torch::Tensor fused_matmul_softmax(torch::Tensor q, torch::Tensor k) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && q.dtype() == torch::kFloat32 &&
                    k.dtype() == torch::kFloat32 && q.dim() == 2 && k.dim() == 2,
                "expected CUDA float32 Q [M, D] and K [N, D]");
    TORCH_CHECK(q.size(1) == k.size(1), "Q and K must share the same D");
    q = q.contiguous();
    k = k.contiguous();
    const int M = static_cast<int>(q.size(0));
    const int N = static_cast<int>(k.size(0));
    const int D = static_cast<int>(q.size(1));
    auto p = torch::empty({q.size(0), k.size(0)}, q.options());
    if (M == 0 || N == 0) {
        return p;
    }
    const int q_stride = (D + 1) & ~1;
    const size_t shared =
        (static_cast<size_t>(q_stride) + static_cast<size_t>(THREADS) * TILE_D) *
        sizeof(float);
    TORCH_CHECK(shared <= 48 * 1024,
                "D is too large for this fused kernel (Q row + K tile must fit in 48 KiB smem)");
    fused_matmul_softmax_kernel<<<M, THREADS, shared, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), p.data_ptr<float>(), M, N, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return p;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_matmul_softmax",
          &fused_matmul_softmax,
          "Fused tiled matmul + online softmax (softmax(Q @ K^T), S not stored)"
    );
}
