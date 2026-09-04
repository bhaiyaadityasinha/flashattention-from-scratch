# CUDA Softmax and Fused Attention

A from-scratch CUDA implementation of several softmax algorithms and a fused `QK^T + softmax` kernel.

The project starts with a basic softmax implementation and progressively adds numerical stability, online normalization, shared-memory tiling, and finally fusion with the matrix multiplication used in attention.

The main goal is to measure the tradeoffs between global memory traffic, numerical stability, synchronization, and GPU data reuse.

## Results

### Softmax

**GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU, Ampere, sm_86, 20 SMs
**Input:** 1024 x 4096 FP32
**CUDA:** 13.0
**PyTorch:** 2.13.0

Each result is the mean of 5 sessions. Every session uses 10 warmup runs followed by 100 timed runs. CUDA events are used for timing.

| Kernel          | Method                       | Time (ms) | Modeled GB/s |
| --------------- | ---------------------------- | --------: | -----------: |
| Naive           | Unstable `exp(x)`            |     0.314 |        160.4 |
| Stable          | Three-pass softmax           |     0.398 |        168.7 |
| Online          | Online normalization         |     0.317 |        158.9 |
| Tiled           | Shared memory + warp shuffle |     0.227 |        147.6 |
| `torch.softmax` | PyTorch reference            |     0.221 |        152.0 |

Modeled bandwidth is based on the theoretical global memory traffic of each implementation. Since the kernels use different numbers of global memory passes, the bandwidth values are not directly comparable. Wall time is the primary performance metric.

The online implementation is about 20% faster than the stable three-pass version. The tiled implementation is close to `torch.softmax`, with measured times of 0.227 ms and 0.221 ms respectively.

The naive implementation is faster than the stable version, but it is not numerically safe. An overflow test using values drawn from N(100, 64) produces non-finite results.

### Fused Matmul + Softmax

The fused kernel computes:

```text
softmax(Q @ K^T)
```

without writing the intermediate score matrix `S` to HBM.

| Kernel          | N=1024, D=64 | N=4096, D=64 |
| --------------- | -----------: | -----------: |
| Fused           |     4.268 ms |    67.149 ms |
| Unfused tiled   |     0.118 ms |     1.432 ms |
| Unfused PyTorch |     0.112 ms |     1.443 ms |

The fused implementation is currently much slower than both unfused versions.

At N=1024 it is 36.2x slower than the unfused tiled implementation. At N=4096 it is 46.9x slower.

This result shows that removing an intermediate tensor from HBM does not automatically make a fused kernel faster. The replacement computation also needs to maintain efficient GPU utilization and data reuse.

## Fused Kernel

The current fused implementation uses one block per query row.

Q is loaded into shared memory and K is processed in tiles. The kernel makes two passes over K.

**Pass 1**

Each thread computes dot products and updates a running `(max, sum)` pair using the online softmax recurrence.

**Pass 2**

The dot products are recomputed and the normalized probabilities are written directly to the output.

The score matrix is never stored in HBM.

### Why It Is Currently Slow

The main limitation is the matrix multiplication decomposition.

A normal tiled GEMM loads tiles of Q and K into shared memory and reuses those values across many output elements. The current fused kernel instead assigns a complete query row to a block and computes dot products with much less reuse of K data.

The kernel also computes every score twice. The first computation is needed for the softmax statistics and the second is needed to produce the final output.

There is synchronization overhead as well. For N=4096, D=64, `THREADS=256`, and `TILE_D=32`, the kernel executes approximately 138 block-wide `__syncthreads()` calls per block.

Of these, 128 come from the two dot-product passes:

```text
16 K panels x 2 D panels x 2 synchronizations x 2 passes
```

The other 10 come from Q loading, the reduction setup, and the 8-step reduction across 256 threads.

These barriers add overhead, but the current benchmark does not establish how much of the slowdown comes from synchronization versus the inefficient GEMM decomposition. Nsight Compute profiling would be needed to separate the two.

At N=1024, the 4 MB score matrix can fit within the GPU's roughly 24 MB L2 cache capacity, which may reduce but does not guarantee elimination of DRAM traffic. At N=4096, the score matrix is 64 MB, so it exceeds the L2 cache capacity and the memory cost of materializing the matrix becomes much more significant.

The next optimization is therefore to redesign the fused kernel around two-dimensional Q and K tiles. This would allow a loaded K tile to be reused across multiple query rows while keeping the online softmax computation.

## Kernel Implementations

### `softmax_naive.cu`

Basic unstable softmax:

```text
exp(x) / sum(exp(x))
```

No maximum is subtracted before evaluating the exponential. This makes it useful as a baseline and as a demonstration of floating-point overflow.

### `softmax_stable.cu`

Three-pass numerically stable softmax:

1. Find the row maximum.
2. Compute `sum(exp(x - max))`.
3. Write the normalized output.

Subtracting the maximum prevents exponential overflow.

### `softmax_online.cu`

Implements the online normalizer described by Milakov and Gimelshein.

Each thread maintains a running `(max, sum)` pair. When a new maximum is encountered, the previous sum is rescaled using:

```text
exp(old_max - new_max)
```

Partial pairs are combined across the block using the same recurrence.

This reduces the statistics calculation to one global memory pass.

### `softmax_tiled.cu`

Loads the entire row into shared memory once.

The reduction uses `__shfl_down_sync` through `warp_reduce_pair`, followed by a small cross-warp reduction.

The output is generated from the shared-memory copy, avoiding another global memory read.

The implementation requires the row to fit within 48 KiB of shared memory.

### `softmax_fused_matmul.cu`

Computes `softmax(Q @ K^T)` without materializing the score matrix.

The implementation uses tiled K loading, online softmax statistics, and a second pass to recompute the scores and write the output.

It is currently a correctness-focused implementation rather than a performance-optimized Flash Attention implementation.

## Correctness

The kernels are tested against PyTorch and several manually constructed cases.

### Random Inputs

Softmax kernels are tested on 512 x 1024 inputs drawn from N(0, 4).

4096 randomly selected output elements are compared against:

```python
torch.softmax(x, dim=-1)
```

The maximum errors are within normal FP32 rounding error, around 1e-7.

### Hand-Checked Example

The online and tiled implementations are tested using:

```text
[1, 2, 3, 0]
```

The result is compared with a closed-form calculation independent of PyTorch.

### Overflow Test

Inputs of size 256 x 2048 are drawn from N(100, 64).

The naive implementation produces non-finite values. Stable, online, and tiled implementations remain finite and match the PyTorch result.

### Fused Kernel Tests

The fused implementation is tested using:

* Random Q and K, 32 x 48 with D=64.
* 4096 sampled output elements compared against `torch.softmax(Q @ K.T)`.
* Non-multiple-of-tile dimensions: 17 x 19 with D=13.
* A hand-constructed case where `Q @ K^T = [1, 2, 3, 0]`.
* Large Q and K values producing scores in the hundreds.

The fused kernel produces finite, numerically correct results across these tests.

## Repository Layout

```text
cuda-softmax/
|
+-- kernels/
|   +-- common.cuh
|   +-- softmax_naive.cu
|   +-- softmax_stable.cu
|   +-- softmax_online.cu
|   +-- softmax_tiled.cu
|   +-- softmax_fused_matmul.cu
|
+-- benchmarks/
|   +-- benchmark_pytorch.py
|
+-- requirements.txt
+-- notes.md
+-- README.md
```

### `kernels/common.cuh`

Shared definitions used by multiple kernels:

* `SoftmaxPair`
* `merge_pair`
* `warp_reduce_pair`

### `kernels/softmax_naive.cu`

Unstable softmax baseline.

### `kernels/softmax_stable.cu`

Three-pass numerically stable softmax.

### `kernels/softmax_online.cu`

Online softmax based on Milakov and Gimelshein.

### `kernels/softmax_tiled.cu`

Shared-memory and warp-shuffle implementation.

### `kernels/softmax_fused_matmul.cu`

Fused `softmax(Q @ K^T)` implementation that does not write the score matrix to HBM.

### `benchmarks/benchmark_pytorch.py`

Builds the CUDA extensions, checks correctness, runs the benchmarks, and compares the results with PyTorch.

### `notes.md`

Development notes containing bugs, implementation decisions, experiments, and benchmark history.

## Build and Run

The project requires PyTorch with CUDA support.

The CUDA kernels are compiled automatically at runtime using `torch.utils.cpp_extension.load`, so no manual `nvcc` command is required.

```bash
pip install -r requirements.txt
python benchmarks/benchmark_pytorch.py
```

## Background

The online softmax implementation follows the recurrence described in:

Milakov, M., & Gimelshein, N. (2018). *Online normalizer calculation for softmax*. arXiv:1805.02867.

The fused attention direction is motivated by:

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and memory-efficient exact attention with IO-awareness*. NeurIPS 2022.

## Next Steps

The current fused implementation proves that the intermediate score matrix can be removed while maintaining numerical correctness.

The main performance work is to change the QK computation from a row-oriented dot-product approach to a two-dimensional tiled GEMM. The goal is to increase reuse of Q and K data, reduce unnecessary global memory traffic, and make the fused computation closer to the structure used by Flash Attention.
