#include <ATen/cuda/CUDAContext.h>
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace py = pybind11;

// FlashAttention-style forward pass for one 2-D attention head.
//
// Q: [M, D], K: [N, D], V: [N, DV] -> O: [M, DV], all row-major float32.
//
// The kernel never materializes S = QK^T or P = softmax(S). It streams K/V tiles while maintaining the online softmax state (m, l) and an unnormalized output accumulator:
//
//   m' = max(m, max(S_tile))
//   l' = exp(m-m') l + sum_j exp(S_j-m')
//   O' = exp(m-m') O + sum_j exp(S_j-m') V_j
//
// The final O / l is equivalent to softmax(QK^T * scale) @ V, up to FP32 rounding.

namespace {
constexpr int kThreads = 128;
constexpr int kTileRows = 16;

__device__ __forceinline__ float warp_sum(float value) {
    constexpr unsigned kMask = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kMask, value, offset);
    }
    return value;
}

__global__ __launch_bounds__(kThreads, 4)
void flash_attention_fwd_kernel(const float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ v,
                                float* __restrict__ out,
                                int m_rows,
                                int n_rows,
                                int d,
                                int dv,
                                float scale,
                                bool causal) {
    extern __shared__ float smem[];
    float* q_tile = smem;
    float* k_tile = q_tile + d;
    float* v_tile = k_tile + kTileRows * d;
    float* scores = v_tile + kTileRows * dv;
    // state[0] = m, state[1] = l, state[2] = rescale of the old accumulator.
    float* state = scores + kTileRows;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (row >= m_rows) {
        return;
    }

    const float* q_row = q + static_cast<long long>(row) * d;
    for (int col = tid; col < d; col += kThreads) {
        q_tile[col] = q_row[col];
    }
    __syncthreads();

    // Each thread owns one output channel; the accumulator stays in a register across all K/V tiles.
    float acc = 0.0f;
    if (tid == 0) {
        state[0] = -FLT_MAX;
        state[1] = 0.0f;
    }
    __syncthreads();

    for (int n0 = 0; n0 < n_rows; n0 += kTileRows) {
        // Coalesced cooperative loads of the K and V tiles.
        for (int idx = tid; idx < kTileRows * d; idx += kThreads) {
            const int r = idx / d;
            const int c = idx - r * d;
            const int n = n0 + r;
            k_tile[idx] = n < n_rows ? k[static_cast<long long>(n) * d + c] : 0.0f;
        }
        for (int idx = tid; idx < kTileRows * dv; idx += kThreads) {
            const int r = idx / dv;
            const int c = idx - r * dv;
            const int n = n0 + r;
            v_tile[idx] = n < n_rows ? v[static_cast<long long>(n) * dv + c] : 0.0f;
        }
        __syncthreads();

        // Each warp computes one QK^T score using shuffle reduction.
#pragma unroll
        for (int offset = 0; offset < kTileRows; offset += 4) {
            const int r = offset + warp;
            const int n = n0 + r;
            float dot = 0.0f;
            if (n < n_rows && (!causal || n <= row)) {
                for (int col = lane; col < d; col += 32) {
                    dot = fmaf(q_tile[col], k_tile[r * d + col], dot);
                }
            }
            dot = warp_sum(dot) * scale;
            if (lane == 0) {
                scores[r] = (n < n_rows && (!causal || n <= row)) ? dot : -FLT_MAX;
            }
        }
        __syncthreads();

        // Thread 0 performs the 16-element online-softmax update.
        if (tid == 0) {
            float tile_m = -FLT_MAX;
#pragma unroll
            for (int r = 0; r < kTileRows; ++r) tile_m = fmaxf(tile_m, scores[r]);
            const float old_m = state[0];
            const float old_l = state[1];
            const float new_m = fmaxf(old_m, tile_m);
            const float alpha = old_l == 0.0f ? 0.0f : expf(old_m - new_m);
            float tile_l = 0.0f;
#pragma unroll
            for (int r = 0; r < kTileRows; ++r) tile_l += expf(scores[r] - new_m);
            state[1] = old_l * alpha + tile_l;
            state[0] = new_m;
            state[2] = alpha;
        }
        __syncthreads();

        const float new_m = state[0];
        if (tid < dv) {
            float next = acc * state[2];
#pragma unroll
            for (int r = 0; r < kTileRows; ++r) {
                next = fmaf(expf(scores[r] - new_m), v_tile[r * dv + tid], next);
            }
            acc = next;
        }
        __syncthreads();
    }

    if (tid < dv) {
        out[static_cast<long long>(row) * dv + tid] = acc / state[1];
    }
}

// D=DV=64 fast path.
// Each block processes 32 query rows and 32 K/V rows per tile.
__global__ __launch_bounds__(256, 2)
void flash_attention_32x32_d64(const float* __restrict__ q, const float* __restrict__ k,
                               const float* __restrict__ v, float* __restrict__ out,
                               int m_rows, int n_rows, float scale, bool causal) {
    extern __shared__ float s[];
    float* qs = s;                 // 32 x 64
    float* ks = qs + 32 * 64;      // 32 x 64
    float* vs = ks + 32 * 64;      // 32 x 64
    float* score = vs + 32 * 64;   // 32 x 32
    float* mx = score + 32 * 32;
    float* den = mx + 32;
    float* alpha = den + 32;
    const int t = threadIdx.x, q0 = blockIdx.x * 32;
    const int qr = t >> 3, vd = (t & 7) * 8;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = t; i < 32 * 64; i += 256) {
        const int r = i >> 6, d = i & 63, row = q0 + r;
        qs[i] = row < m_rows ? q[static_cast<long long>(row) * 64 + d] : 0.0f;
    }
    if (t < 32) { mx[t] = -FLT_MAX; den[t] = 0.0f; }
    __syncthreads();
    for (int n0 = 0; n0 < n_rows; n0 += 32) {
        for (int i = t; i < 32 * 64; i += 256) {
            const int r = i >> 6, d = i & 63, row = n0 + r;
            ks[i] = row < n_rows ? k[static_cast<long long>(row) * 64 + d] : 0.0f;
            vs[i] = row < n_rows ? v[static_cast<long long>(row) * 64 + d] : 0.0f;
        }
        __syncthreads();
        // Compute the 32x32 QK^T score tile from shared-memory Q and K.
        const int kb = t & 31, qbase = t >> 5;
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            const int r = qbase + group * 8, row = q0 + r, key = n0 + kb;
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < 64; ++d) dot = fmaf(qs[r * 64 + d], ks[kb * 64 + d], dot);
            score[r * 32 + kb] = (row < m_rows && key < n_rows && (!causal || key <= row))
                                   ? dot * scale : -FLT_MAX;
        }
        __syncthreads();
        if (t < 32) {
            float tm = -FLT_MAX;
#pragma unroll
            for (int j = 0; j < 32; ++j) tm = fmaxf(tm, score[t * 32 + j]);
            const float nm = fmaxf(mx[t], tm);
            const float a = den[t] == 0.0f ? 0.0f : expf(mx[t] - nm);
            float dl = 0.0f;
#pragma unroll
            for (int j = 0; j < 32; ++j) dl += expf(score[t * 32 + j] - nm);
            mx[t] = nm; den[t] = den[t] * a + dl; alpha[t] = a;
        }
        __syncthreads();
        if (q0 + qr < m_rows) {
#pragma unroll
            for (int x = 0; x < 8; ++x) {
                float next = acc[x] * alpha[qr];
#pragma unroll
                for (int j = 0; j < 32; ++j)
                    next = fmaf(expf(score[qr * 32 + j] - mx[qr]), vs[j * 64 + vd + x], next);
                acc[x] = next;
            }
        }
        __syncthreads();
    }
    if (q0 + qr < m_rows)
#pragma unroll
        for (int x = 0; x < 8; ++x) out[static_cast<long long>(q0 + qr) * 64 + vd + x] = acc[x] / den[qr];
}

}  // namespace

torch::Tensor flash_attention_fwd(torch::Tensor q,
                                  torch::Tensor k,
                                  torch::Tensor v,
                                  double softmax_scale = 0.0,
                                  bool causal = false) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q, K, and V must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32 &&
                    v.dtype() == torch::kFloat32,
                "expected float32 Q, K, and V");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2,
                "expected Q [M, D], K [N, D], and V [N, DV]");
    TORCH_CHECK(q.size(1) == k.size(1), "Q and K must share D");
    TORCH_CHECK(k.size(0) == v.size(0), "K and V must share N");

    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    const int m_rows = static_cast<int>(q.size(0));
    const int n_rows = static_cast<int>(k.size(0));
    const int d = static_cast<int>(q.size(1));
    const int dv = static_cast<int>(v.size(1));
    TORCH_CHECK(d > 0 && d <= 128 && dv > 0 && dv <= 128,
                "this FP32 kernel supports 1 <= D, DV <= 128");
    TORCH_CHECK(m_rows >= 0 && n_rows >= 0, "tensor dimensions must fit in int");

    auto out = torch::empty({q.size(0), v.size(1)}, q.options());
    if (m_rows == 0) return out;
    if (n_rows == 0) return torch::zeros_like(out);

    const float scale = softmax_scale == 0.0
                            ? 1.0f / sqrtf(static_cast<float>(d))
                            : static_cast<float>(softmax_scale);
    if (d == 64 && dv == 64) {
        constexpr size_t fast_shared = (32 * 64 * 3 + 32 * 32 + 32 * 3) * sizeof(float);
        flash_attention_32x32_d64<<<(m_rows + 31) / 32, 256, fast_shared,
                                     at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
            m_rows, n_rows, scale, causal);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    const size_t shared = (static_cast<size_t>(d) +
                           static_cast<size_t>(kTileRows) * (d + dv + 1) + 3) * sizeof(float);
    TORCH_CHECK(shared <= 48 * 1024, "FlashAttention tile exceeds 48 KiB shared memory");
    flash_attention_fwd_kernel<<<m_rows, kThreads, shared, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
        m_rows, n_rows, d, dv, scale, causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_fwd", &flash_attention_fwd,
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("softmax_scale") = 0.0, py::arg("causal") = false,
          "FlashAttention-style FP32 forward pass (QK^T softmax V; no S/P materialization)");
}
