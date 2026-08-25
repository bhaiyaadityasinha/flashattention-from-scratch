#include "common.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Single-pass online softmax (paper Algorithm 3 + parallel merge, eq. 4).
// Each thread walks its strided slice once, keeping a running (max, sum)
// pair. Those pairs are then merged across the block with the same rule:
//   m' = max(m_a, m_b)
//   d' = d_a * exp(m_a - m') + d_b * exp(m_b - m')
// The output write still rereads the row from global memory (3 accesses
// per element: 1 stats load + 1 write-pass load + 1 store).
__global__ void softmax_online_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int rows,
                                      int cols) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int tid = threadIdx.x;
    const float* row_in = input + static_cast<long long>(row) * cols;
    float* row_out = output + static_cast<long long>(row) * cols;

    extern __shared__ SoftmaxPair partial[];

    SoftmaxPair local = empty_pair();
    for (int col = tid; col < cols; col += blockDim.x) {
        // Sequential online update for one element: merge with (x, 1).
        local = merge_pair(local, {row_in[col], 1.0f});
    }
    partial[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] = merge_pair(partial[tid], partial[tid + stride]);
        }
        __syncthreads();
    }
    const SoftmaxPair total = partial[0];
    const float inv_d = 1.0f / total.d;

    for (int col = tid; col < cols; col += blockDim.x) {
        row_out[col] = expf(row_in[col] - total.m) * inv_d;
    }
}

torch::Tensor softmax_online(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32 && input.dim() == 2,
                "expected a CUDA float32 [rows, cols] tensor");
    input = input.contiguous();
    auto output = torch::empty_like(input);
    constexpr int threads = 256;
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    softmax_online_kernel<<<rows, threads, threads * sizeof(SoftmaxPair),
                            at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax_online, "Online CUDA softmax");
}
