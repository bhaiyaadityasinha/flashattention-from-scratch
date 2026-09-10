# FlashAttention From Scratch — Technical Report

## 1. Overview

This project implements a from-scratch CUDA **FlashAttention-style exact attention forward pass**, supported by a progression of softmax kernels and attention-fusion experiments.

**This is a research/learning implementation, not a production replacement for FlashAttention.** The goal is to study attention execution through implementation, measurement, failed experiments, and hardware-guided optimization.

The work explores how numerical stability, memory traffic, data reuse, tiling, synchronization, and GPU execution structure interact in attention workloads.

The central GPU-systems question is:

> Does eliminating global-memory traffic improve performance if the replacement computation has poor data reuse or an inefficient execution structure?

The project progressed from standalone softmax implementations, through an unsuccessful row-oriented `QK^T + softmax` fusion, to a tiled FlashAttention-style forward pass that avoids materializing the full attention matrices in global memory.

The key engineering lesson is that **reducing memory traffic alone is not sufficient**. The computation must also be organized around effective data reuse and an execution structure suited to the GPU.

## 2. Experimental environment

* NVIDIA GeForce RTX 3050 6GB Laptop GPU
* Ampere, `sm_86`, 20 SMs
* CUDA 13.0
* PyTorch 2.13.0
* FP32

Softmax results are the mean of 5 sessions, with 10 warmup runs and 100 timed runs per session using CUDA events.

Modeled bandwidth values are calculated from theoretical global-memory traffic. They are **not measured DRAM throughput**.

## 3. Supporting softmax progression

The project began with four CUDA softmax implementations:

* naive
* three-pass stable
* online
* shared-memory tiled

The stable implementation performs row maximum, shifted exponential sum, and normalization. The online implementation maintains a running `(max, sum)` pair. The tiled implementation loads the row into shared memory and uses warp-shuffle reduction plus a cross-warp reduction.

### Softmax benchmark

Input: `1024 × 4096` FP32.

| Kernel    | Time (ms) | Modeled GB/s |
| --------- | --------: | -----------: |
| Naive     |     0.323 |        155.9 |
| Stable    |     0.397 |        169.2 |
| Online    |     0.316 |        159.7 |
| **Tiled** | **0.231** |        145.6 |
| PyTorch   |     0.227 |        148.2 |

The online implementation is about **20% faster** than the stable three-pass implementation on this benchmark. The tiled implementation is within roughly **1.8%** of the PyTorch reference in wall time.

These softmax experiments provide the numerical-stability and reduction groundwork for the later attention kernels.

## 4. Softmax correctness

Random `512 × 1024` inputs are drawn from `N(0, 4)`, with 4096 sampled outputs compared against `torch.softmax`. Typical maximum error is around `1e-7`.

An additional hand-checked case uses `[1, 2, 3, 0]`.

For overflow stress, `256 × 2048` inputs are drawn from `N(100, 64)`. The naive implementation produces non-finite values, while the stable, online, and tiled variants remain finite in the tested cases.

## 5. First attention experiment: fused `QK^T + softmax`

The first attention kernel attempted to compute:

```text
softmax(Q @ K^T)
```

without writing the full score matrix to global memory.

| Kernel                 | N=1024, D=64 | N=4096, D=64 |
| ---------------------- | -----------: | -----------: |
| Fused `QK^T + softmax` |     4.249 ms |    66.457 ms |
| Unfused tiled CUDA     |     0.116 ms |     1.425 ms |
| Unfused PyTorch        |     0.118 ms |     1.445 ms |

Relative to the unfused tiled CUDA baseline, the fused kernel is **36.6× slower** at `N=1024` and **46.6× slower** at `N=4096`.

This is the central negative result of the project:

> Eliminating an intermediate global-memory write does not automatically produce a faster kernel.

### Execution structure

The fused kernel uses one block per query row. Q is loaded into shared memory, K is processed in tiles, and two passes are made over K:

1. Compute scores and update online softmax statistics.
2. Recompute scores and write the normalized output.

The score matrix is never written to global memory.

### Synchronization accounting

For `N=4096`, `D=64`, `THREADS=256`, and `TILE_D=32`, the kernel executes approximately **138 block-wide `__syncthreads()` calls per block**.

The dot-product passes account for:

```text
16 K panels × 2 D panels × 2 synchronizations × 2 passes = 128
```

The remaining approximately 10 barriers come from Q loading, reduction setup, and the 8-step reduction across 256 threads.

These barriers are present, but the benchmark does not establish their relative contribution to the slowdown versus the row-oriented GEMM decomposition. Nsight Compute is required to separate those effects.

### Data reuse and cache considerations

A conventional tiled GEMM reuses loaded Q and K tiles across many output elements. The original fused kernel assigns one query row to each block, limiting cross-query reuse of K.

At `N=1024`, the 4 MB score matrix is smaller than the GPU's reported L2 capacity, so some score-related traffic in the unfused path may be served from cache rather than DRAM. This does not guarantee residency or identify the dominant bottleneck.

At `N=4096`, the score matrix is 64 MB, making persistent L2 residency impossible.

These observations do not by themselves identify the dominant bottleneck.

## 6. Tiled FlashAttention-style forward pass

The failed fusion experiment motivated a redesign around **cross-query data reuse and tiled attention computation**.

The redesigned kernel computes the exact attention operation:

```text
O = softmax(Q @ K^T / sqrt(D)) @ V
```

without materializing the full score or probability matrices in global memory.

It processes K/V tiles sequentially while maintaining a running maximum `m`, normalization factor `l`, and unnormalized output accumulator.

For each tile:

```text
m' = max(m, max(S_tile))

l' = exp(m - m') * l
     + sum_j exp(S_j - m')

O' = exp(m - m') * O
     + sum_j exp(S_j - m') V_j
```

The final result is:

```text
O / l
```

up to normal FP32 rounding.

This is **FlashAttention-style** rather than a claim of reproducing the production FlashAttention implementation.

### D=DV=64 fast path

The benchmarked fast path processes 32 query rows per block and 32 K/V rows per tile.

Shared memory contains:

```text
Q tile:      32 × 64
K tile:      32 × 64
V tile:      32 × 64
score tile:  32 × 32
```

A loaded K/V tile is reused across multiple query rows. Q remains in shared memory and the output accumulator remains in registers.

The 64-element Q·K dot product is computed directly in a thread rather than through a warp-level reduction.

A general FP32 kernel supports `1 <= D, DV <= 128`, subject to the shared-memory limit.

## 7. FlashAttention-style benchmark

The comparison performs the same overall attention computation:

```text
softmax(Q @ K^T / sqrt(D)) @ V
```

| Kernel                   | N=1024, D=64 |  N=4096, D=64 |
| ------------------------ | -----------: | ------------: |
| Unfused attention        |     4.309 ms |     67.708 ms |
| **FlashAttention-style** | **0.760 ms** | **10.322 ms** |
| PyTorch SDPA             |     0.174 ms |      2.255 ms |

The FlashAttention-style implementation is:

* **5.7× faster** than the unfused attention baseline at `N=1024`.
* **6.6× faster** at `N=4096`.

The gap to PyTorch SDPA decreases to roughly:

* **4.4×** at `N=1024`.
* **4.6×** at `N=4096`.

PyTorch SDPA selects its own native backend, so this is a **reference comparison rather than a controlled implementation-to-implementation comparison**.

The benchmark therefore demonstrates a substantial performance improvement after changing the execution structure and increasing cross-query reuse. It does not isolate the contribution of any individual optimization.

## 8. Interpreting the improvement

The most direct structural change is execution granularity.

The original fused kernel uses one query row per block. The redesigned `D=DV=64` kernel processes 32 query rows together, allowing a loaded K/V tile to be reused across multiple queries.

The redesigned kernel also avoids writing complete score and probability matrices to global memory. Score tiles are consumed on-chip to update the online softmax state and output accumulator.

The measured result is a **5.7–6.6× reduction in execution time relative to the unfused attention baseline**.

The benchmark does not isolate the contribution of individual optimizations, so the speedup should not be attributed to memory-traffic reduction alone.

## 9. Remaining gap to PyTorch SDPA

The FlashAttention-style implementation remains roughly **4.4–4.6× slower** than PyTorch SDPA on the tested shapes.

No single cause is assigned without profiling.

Possible contributors include:

* scalar FP32 FMA instead of Tensor Core / MMA execution,
* lack of warp-specialized execution,
* lack of asynchronous global-to-shared-memory pipelines,
* non-optimal architecture-specific tile and thread mappings,
* register/shared-memory allocation and scheduling choices,
* instruction-level optimizations used by production attention kernels.

The implementation has **not yet been profiled with Nsight Compute**, so the relative contribution of these factors is unknown.

The next optimization cycle should therefore be hardware-guided rather than based on guessing at individual bottlenecks.

## 10. Correctness

The fused kernel is tested with random Q/K inputs, non-multiple-of-tile dimensions, hand-constructed cases, and large logits.

The FlashAttention-style kernel is tested with random Q/K/V inputs, non-multiple-of-tile dimensions (`D=13`, `DV=11`), hand-constructed cases, and large-logit stress tests.

For the FlashAttention-style large-logit stress test, the measured maximum error ranged from `2.4e-7` to `4.1e-4` across sessions.

This is larger and more variable than the roughly `1e-7` errors observed for the standalone softmax kernels. The current benchmark does not isolate the source of this variation.

## 11. Benchmark limitations

* Results are measured on one RTX 3050 Laptop GPU.
* Performance can vary with GPU architecture and operating conditions.
* Softmax GB/s values are modeled from theoretical traffic, not measured DRAM throughput.
* PyTorch SDPA selects its own backend.
* The tested attention shapes are limited.
* No Nsight Compute profiling has yet been performed.
* The measurements do not isolate individual low-level optimizations.

## 12. Engineering lessons

### Fewer memory operations are not automatically faster

The first fused kernel removed an intermediate global-memory write but became tens of times slower. The experiment exposed execution structure as an optimization constraint.

### Data reuse is a first-class design variable

The tiled attention kernel increases reuse of K/V tiles across multiple query rows. This structural change accompanies the large measured speedup.

### Fusion changes the computational problem

Fusing operations changes how work is partitioned, synchronized, and reused; it is not simply a matter of combining two kernels.

### Measure before assigning a bottleneck

The synchronization count and GEMM structure provide plausible explanations for the original fused kernel's behavior, but the benchmark does not distinguish their relative impact.

## 13. Next steps

The next optimization cycle should start with Nsight Compute.

The goal is to determine whether the remaining gap is dominated by instruction throughput, memory movement, synchronization, register pressure, occupancy, shared-memory behavior, or the absence of Tensor Core/MMA execution.

Potential follow-up experiments:

1. Tensor Core / MMA-based dot products.
2. Warp-specialized producer/consumer execution.
3. Asynchronous global-to-shared-memory pipelines.
4. Architecture-specific tile and thread mappings.
5. Register and shared-memory tuning.
6. Shape-specialized kernels.

The objective is not simply to reproduce a production attention kernel, but to understand **why different GPU execution structures produce radically different performance for the same mathematical operation**.

## 14. Related work

Milakov, M. & Gimelshein, N. (2018). *Online normalizer calculation for softmax*. arXiv:1805.02867.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
