# FlashAttention From Scratch

A from-scratch CUDA implementation study of **FlashAttention-style exact attention**, covering numerical stability, memory traffic, tiling, data reuse, and GPU execution structure.

> **This is a learning/research implementation, not a production replacement for FlashAttention.** The goal is to study the execution structure behind IO-aware attention through implementation, measurement, and failed experiments.

The project starts with standalone softmax kernels, explores a failed `QKᵀ + softmax` fusion, and culminates in a tiled FlashAttention-style forward pass that avoids materializing the full attention matrices in global memory.

> **Core finding:** reducing global-memory traffic alone is not enough. High-performance GPU kernels also require effective data reuse and an execution structure matched to the hardware.

## Results at a glance

**NVIDIA RTX 3050 6GB Laptop · Ampere (`sm_86`) · FP32 · CUDA 13.0**

### FlashAttention-style attention — `N × N`, `D=DV=64`

| Implementation           |       N=1024 |        N=4096 |
| ------------------------ | -----------: | ------------: |
| Unfused attention        |     4.309 ms |     67.708 ms |
| **FlashAttention-style** | **0.760 ms** | **10.322 ms** |
| PyTorch SDPA             |     0.174 ms |      2.255 ms |

The FlashAttention-style kernel is **5.7× faster** than the unfused attention baseline at `N=1024` and **6.6× faster** at `N=4096`.

The gap to PyTorch SDPA falls from roughly **25× → 4.4×** at `N=1024` and **30× → 4.6×** at `N=4096`.

These measurements demonstrate the effect of changing the execution structure and increasing cross-query reuse, but do **not** isolate the contribution of individual optimizations.

PyTorch SDPA selects its own native backend, so this is a **reference comparison rather than a controlled implementation-to-implementation comparison**. Its backend may use **Tensor Core/MMA-based matrix multiplication**, among other architecture-specific optimizations.

### Softmax — `1024 × 4096`

| Implementation |         Time |
| -------------- | -----------: |
| Naive          |     0.323 ms |
| Stable         |     0.397 ms |
| Online         |     0.316 ms |
| **Tiled**      | **0.231 ms** |
| PyTorch        |     0.227 ms |

The tiled softmax kernel is within **1.8%** of `torch.softmax` on this workload. Online softmax is about **20% faster** than the stable three-pass implementation.

## The FlashAttention-style kernel

The final kernel computes:

```text
O = softmax(Q @ Kᵀ / √D) @ V
```

without materializing the **full score (`S`) or probability (`P`) matrices in global memory**.

It maintains the online softmax statistics and output accumulator across K/V tiles:

```text
m' = max(m, max(S_tile))

l' = exp(m - m') * l
     + sum_j exp(S_j - m')

O' = exp(m - m') * O
     + sum_j exp(S_j - m') V_j
```

The final output is:

```text
O / l
```

### D=DV=64 fast path

The benchmarked fast path processes **32 query rows per block** and 32 K/V rows per tile:

```text
Q tile:      32 × 64
K tile:      32 × 64
V tile:      32 × 64
score tile:  32 × 32
```

Q, K, V, and the score tile reside in **shared memory**, while the **output accumulator remains in registers**.

Each K/V tile is reused across multiple query rows. The 64-element Q·K dot product is computed directly in a thread rather than through a warp-level reduction.

A separate general FP32 kernel supports `1 <= D, DV <= 128`, subject to shared-memory limits.

## The experiment that motivated the redesign

Before the tiled attention kernel, I attempted to fuse:

```text
softmax(Q @ Kᵀ)
```

into a single row-oriented kernel without writing the full score matrix to global memory.

| Implementation        |   N=1024 |    N=4096 |
| --------------------- | -------: | --------: |
| Fused `QKᵀ + softmax` | 4.249 ms | 66.457 ms |
| Unfused tiled         | 0.116 ms |  1.425 ms |
| Unfused PyTorch       | 0.118 ms |  1.445 ms |

The fused kernel was **36.6×** slower than the unfused tiled implementation at `N=1024` and **46.6×** slower at `N=4096`.

The experiment showed that eliminating an intermediate global-memory write does not automatically make a kernel faster.

The design used:

* One block per query row.
* Q in shared memory.
* K processed in tiles.
* Two passes over K.
* Online softmax statistics.
* Score recomputation.
* No full score matrix written to global memory.

However, it provided limited cross-query reuse of K data, recomputed every score, and introduced substantial synchronization. For `N=4096`, `D=64`, `THREADS=256`, and `TILE_D=32`, it executes approximately **138 `__syncthreads()` calls per block**.

The benchmark does not establish whether synchronization or the matrix-multiplication decomposition is the dominant bottleneck; **Nsight Compute profiling has not yet been performed**.

This negative result led to the final redesign around **tiled attention and cross-query K/V reuse**.

## Why the PyTorch gap remains

The FlashAttention-style implementation remains roughly **4.4–4.6× slower** than PyTorch SDPA.

The benchmark does not establish a single cause. Potential contributors include:

* Tensor Core / MMA-based computation.
* Warp-specialized execution.
* Asynchronous global-to-shared-memory pipelines.
* Architecture-specific tile and thread mappings.
* Register and shared-memory management.
* Low-level scheduling and instruction-level optimization.

The implementation has **not yet been profiled with Nsight Compute**. The next optimization cycle should therefore be hardware-guided rather than based on guessing at individual bottlenecks.

## Correctness

The kernels are tested against PyTorch using random inputs, non-multiple-of-tile dimensions, hand-constructed cases, and large-logit stress tests.

Standalone softmax tests typically produce maximum errors around **`1e-7`**.

For overflow stress using inputs drawn from `N(100, 64)`, the naive implementation produces non-finite values while the stable, online, and tiled implementations remain finite.

For the FlashAttention-style large-logit stress test, the measured maximum error ranged from **`2.4e-7` to `4.1e-4`** across sessions. This is larger and more variable than the roughly `1e-7` errors observed for the standalone softmax kernels. The current tests do not isolate the source of this difference.

## Softmax progression

The attention kernel builds on several progressively optimized softmax implementations:

* `softmax_naive.cu` — unstable baseline.
* `softmax_stable.cu` — three-pass numerically stable softmax.
* `softmax_online.cu` — online normalization.
* `softmax_tiled.cu` — shared-memory + warp-shuffle softmax.

These implementations provide the numerical-stability and reduction groundwork used when reasoning about the attention kernels.

## Repository layout

```text
flashattention-from-scratch/
├── kernels/
│   ├── common.cuh
│   ├── softmax_naive.cu
│   ├── softmax_stable.cu
│   ├── softmax_online.cu
│   ├── softmax_tiled.cu
│   ├── softmax_fused_matmul.cu
│   └── flash_attention_fwd.cu
├── benchmarks/
│   └── benchmark_pytorch.py
├── requirements.txt
├── notes.md
├── README.md
└── REPORT.md
```

### Attention kernel

`flash_attention_fwd.cu` implements the tiled FlashAttention-style forward pass:

```text
O = softmax(Q @ Kᵀ / √D) @ V
```

The full `S` and `P` matrices are not materialized in global memory.

### Fusion experiment

`softmax_fused_matmul.cu` contains the unsuccessful row-oriented fusion experiment that motivated the tiled redesign.

## Benchmark methodology

* NVIDIA RTX 3050 6GB Laptop GPU
* Ampere, `sm_86`
* FP32
* CUDA 13.0
* PyTorch 2.13.0
* 5 benchmark sessions
* 10 warmup + 100 timed runs per session
* CUDA events for timing

Softmax GB/s values are **modeled from theoretical global-memory traffic**, not measured DRAM bandwidth. Wall time is the primary performance metric.

## Run

```bash
pip install -r requirements.txt
python benchmarks/benchmark_pytorch.py
```

CUDA extensions are compiled automatically through `torch.utils.cpp_extension.load`.

## Research questions

* How does data reuse affect GPU attention performance?
* When does reducing global-memory traffic actually improve runtime?
* How should online softmax be integrated with tiled matrix multiplication?
* What execution structures make kernel fusion effective?
* How much performance is left in a hand-written FP32 implementation without Tensor Core/MMA execution?
* Which hardware-level changes matter most after the algorithmic structure is correct?

## Current status

The project now has a working **FlashAttention-style tiled forward pass** that substantially outperforms the original fused attention design on the tested workloads.

Relative to the **unfused attention baseline**, the gap to PyTorch SDPA falls from roughly **25–30× to 4.4–4.6×**.

The next optimization cycle should begin with **Nsight Compute profiling** to identify the actual hardware bottlenecks before introducing lower-level optimizations such as:

* Tensor Core / MMA-based dot products.
* Warp-specialized producer/consumer execution.
* Asynchronous global-to-shared-memory pipelines.
* Architecture-specific tile and thread mappings.
* Register and shared-memory optimization.
* Additional shape-specialized kernels.

The objective is not simply to reproduce a production attention kernel. It is to understand, through **implementation, measurement, failed experiments, and hardware-guided optimization**, why different GPU execution structures produce radically different performance for the same mathematical operation.

## Detailed report

See [`REPORT.md`](REPORT.md) for the full implementation discussion, synchronization analysis, correctness methodology, benchmark history, limitations, and optimization reasoning.

## References

* Milakov, M. & Gimelshein, N. (2018). *Online normalizer calculation for softmax*. arXiv:1805.02867.
* Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
