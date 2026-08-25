#include "common.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Tiled online softmax: one row per block.
// 1. Cooperatively load the row into shared memory (one global read).
// 2. Each thread builds a local (max, sum) pair from its tile slice.
// 3. Warp-shuffle reduction of those pairs (__shfl_down_sync).
// 4. One warp merges the per-warp results from a small shared array.
// 5. Write softmax from the shared-memory copy (no second global read).
//
// Requires the row to fit in dynamic shared memory (see host check).
__global__ void softmax_tiled_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int rows,
                                     int cols) {
    extern __shared__ unsigned char storage[];
    float* tile = reinterpret_cast<float*>(storage);
    // Align warp partials to 8 bytes after an odd-length float tile.
    const size_t tile_bytes = (static_cast<size_t>(cols) * sizeof(float) + 7u) & ~size_t{7};
    SoftmaxPair* warp_partials = reinterpret_cast<SoftmaxPair*>(storage + tile_bytes);

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int tid = threadIdx.x;
    const float* row_in = input + static_cast<long long>(row) * cols;
    float* row_out = output + static_cast<long long>(row) * cols;

    for (int col = tid; col < cols; col += blockDim.x) {
        tile[col] = row_in[col];
    }
    __syncthreads();

    SoftmaxPair local = empty_pair();
    for (int col = tid; col < cols; col += blockDim.x) {
        local = merge_pair(local, {tile[col], 1.0f});
    }
    local = warp_reduce_pair(local);

    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_partials[warp] = local;
    }
    __syncthreads();

    const int nwarps = blockDim.x >> 5;
    if (warp == 0) {
        SoftmaxPair total = (lane < nwarps) ? warp_partials[lane] : empty_pair();
        total = warp_reduce_pair(total);
        if (lane == 0) {
            warp_partials[0] = total;
        }
    }
    __syncthreads();
    const SoftmaxPair total = warp_partials[0];
    const float inv_d = 1.0f / total.d;

    for (int col = tid; col < cols; col += blockDim.x) {
        row_out[col] = expf(tile[col] - total.m) * inv_d;
    }
}

torch::Tensor softmax_tiled(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32 && input.dim() == 2,
                "expected a CUDA float32 [rows, cols] tensor");
    input = input.contiguous();
    auto output = torch::empty_like(input);
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    const size_t tile_bytes = (static_cast<size_t>(cols) * sizeof(float) + 7u) & ~size_t{7};
    const size_t shared = tile_bytes + warps * sizeof(SoftmaxPair);
    TORCH_CHECK(shared <= 48 * 1024,
                "row is too wide for this single-tile kernel (need cols that fit in 48 KiB smem)");
    softmax_tiled_kernel<<<rows, threads, shared, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax_tiled, "Tiled warp-shuffle CUDA softmax");
}
