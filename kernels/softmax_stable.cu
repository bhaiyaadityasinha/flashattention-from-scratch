#include <ATen/cuda/CUDAContext.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Numerically stable softmax (paper Algorithm 2): three global-memory passes.
//   1. max over the row
//   2. sum of exp(x - max)
//   3. write exp(x - max) / sum
// Subtracting the row max makes every exponent ≤ 0, so nothing overflows,
// and the result is identical because num and den are both scaled by e^{-max}.
__global__ void softmax_stable_kernel(const float* __restrict__ input,
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

    extern __shared__ float scratch[];

    // Pass 1: row max. Idle threads contribute -FLT_MAX, the identity of max.
    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_in[col]);
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    const float row_max = scratch[0];

    // Pass 2: sum of exp(x - max). Idle threads contribute 0.
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += expf(row_in[col] - row_max);
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / scratch[0];

    // Pass 3: normalize. Third load of the row from global memory.
    for (int col = tid; col < cols; col += blockDim.x) {
        row_out[col] = expf(row_in[col] - row_max) * inv_sum;
    }
}

torch::Tensor softmax_stable(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32 && input.dim() == 2,
                "expected a CUDA float32 [rows, cols] tensor");
    input = input.contiguous();
    auto output = torch::empty_like(input);
    constexpr int threads = 256;
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    softmax_stable_kernel<<<rows, threads, threads * sizeof(float),
                            at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax_stable, "Three-pass stable CUDA softmax");
}
