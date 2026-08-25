#include <ATen/cuda/CUDAContext.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Intentionally unstable softmax (paper Algorithm 1).
// No max subtraction. expf(x) overflows float32 for x ≳ 88.7.
//
// Each block owns one row. Threads stride over columns, then combine
// partial sums with a shared-memory tree reduction (no warp shuffles).
__global__ void softmax_unstable_kernel(const float* __restrict__ input,
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

    // Pass 1: sum of exp(x) — this is the overflow path.
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += expf(row_in[col]);
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float denom = scratch[0];

    // Pass 2: write exp(x) / sum. If denom is inf, this produces 0 or nan.
    for (int col = tid; col < cols; col += blockDim.x) {
        row_out[col] = expf(row_in[col]) / denom;
    }
}

torch::Tensor softmax_unstable(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32 && input.dim() == 2,
                "expected a CUDA float32 [rows, cols] tensor");
    input = input.contiguous();
    auto output = torch::empty_like(input);
    constexpr int threads = 256;
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    softmax_unstable_kernel<<<rows, threads, threads * sizeof(float),
                              at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax_unstable, "Unstable CUDA softmax (no max subtraction)");
}
