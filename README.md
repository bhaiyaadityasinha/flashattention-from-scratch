# CUDA Softmax

A from-scratch CUDA softmax project that implements four variants — unstable,
stable, online, and tiled — demonstrating the progression from a numerically
broken baseline to a kernel that matches PyTorch's `torch.softmax` in wall time.

The project implements the online normalizer algorithm from Milakov &
Gimelshein (2018) and demonstrates its core claim empirically: eliminating
one global memory pass reduces wall time by ~27% on real hardware.

---

## Benchmark results

**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere, sm\_86,
20 SMs, 6.4 GB) · 1024×4096 FP32
**Software:** CUDA 13.3 · PyTorch 2.x · `-O3 --use_fast_math`
**Methodology:** mean of 50 consecutive sessions (10 warmup + 100 timed
runs each) · all kernels built via `torch.utils.cpp_extension.load`

| Kernel | Method | Time (ms) | Modeled GB/s |
|---|---|---|---|
| Naive | Unstable `exp(x)`, no max subtraction | 0.333 | 151.0 |
| Stable | Three global passes, max subtraction | 0.403 | 166.7 |
| Online | Running `(max, sum)` pairs, one stats pass | 0.318 | 158.5 |
| Tiled | Shared-memory row + `__shfl_down_sync` | 0.225 | 149.0 |
| torch.softmax | Vendor kernel (reference) | 0.224 | 224.8 |

*Modeled bandwidth is computed from the paper's theoretical access counts
(naive/online = 3N, stable = 4N, tiled = 2N bytes) and is not directly
comparable across kernels — PyTorch likely uses async memory operations
internally. Wall time is the primary metric.*

**Key findings:**

- Online is 27% faster than stable (0.318 ms vs 0.403 ms), directly
  confirming that eliminating one global memory pass reduces wall time on
  real hardware — the central claim of Milakov & Gimelshein.
- Tiled (0.225 ms) matches `torch.softmax` (0.224 ms) within measurement
  noise — a difference of 0.001 ms across fifety sessions.
- Naive is faster than stable (0.333 ms vs 0.403 ms) despite being
  numerically incorrect. Without max subtraction, fewer `expf` calls are
  needed and shared memory operations are simpler — arithmetic is lighter
  even though the result overflows for large inputs.

---

## Kernel variants

**Naive (`softmax_naive.cu`)**
No max subtraction. Computes `exp(x)` directly and divides by the sum.
`expf` overflows float32 for inputs above ~88.7, producing `inf` or `nan`
for rows with large values. Included as an explicit control case that
demonstrates the overflow failure mode — verified to produce non-finite
outputs on inputs drawn from N(100, 64).

**Stable (`softmax_stable.cu`)**
Three global memory passes: (1) row max, (2) sum of `exp(x − max)`,
(3) write `exp(x − max) / sum`. Subtracting the row max before
exponentiation guarantees all exponents are ≤ 0, so nothing overflows.
The output pass re-reads the row from global memory a third time.

**Online (`softmax_online.cu`)**
Single-pass statistics using the Milakov & Gimelshein recurrence
(Algorithm 3 + parallel merge, eq. 4). Each thread maintains a running
`(max, sum)` pair; when a new element is seen with a larger value, the
running sum is rescaled by `exp(old_max − new_max)` before the new term
is added. Partial pairs are merged across the block with the same rule.
Eliminates one global memory pass compared to stable. Output still reads
the row a second time.

**Tiled (`softmax_tiled.cu`)**
Cooperatively loads the entire row into shared memory once. Statistics
are computed from shared memory via `warp_reduce_pair` (`__shfl_down_sync`
reduction of `SoftmaxPair`); one warp merges per-warp results from a
small shared array. Output is written from the shared-memory copy —
no second read from global memory. Reduces to 2 global memory passes
(1 read + 1 write). Requires the row to fit in 48 KiB of shared memory
(checked at launch). Matches `torch.softmax` wall time on this hardware.

---

## Correctness

Each kernel is verified at three levels:

1. **Random inputs** — 4096 randomly sampled elements checked against
   `torch.softmax` on 512×1024 inputs drawn from N(0, 4). All errors
   within 1e-7 to 4e-7 (normal single-precision rounding).

2. **Hand example** — `[1, 2, 3, 0]` checked against a closed-form
   expected value for online and tiled, independent of PyTorch.

3. **Overflow stress** — 256×2048 inputs drawn from N(100, 64).
   Naive produces non-finite outputs (verified). Stable, online, and
   tiled all produce finite results matching `torch.softmax`.

---

## Repository layout

'''
kernels/
  common.cuh                   SoftmaxPair, merge_pair, warp_reduce_pair
  softmax_naive.cu             Unstable baseline (overflow control case)
  softmax_stable.cu            Three-pass numerically stable softmax
  softmax_online.cu            Online single-pass softmax (Milakov & Gimelshein)
  softmax_tiled.cu             Tiled + warp-shuffle softmax
benchmarks/
  benchmark_pytorch.py         Build, verify, and time all kernels vs PyTorch
requirements.txt
notes.md                       Development log — bugs, decisions, benchmark history
'''

---

## Build and run

Requires PyTorch with CUDA support. Kernels are compiled via
`torch.utils.cpp_extension.load` at runtime — no manual `nvcc` invocation needed.

```bash
pip install -r requirements.txt
python benchmarks/benchmark_pytorch.py
```

Optional arguments (rows, cols, iterations):

```bash
python benchmarks/benchmark_pytorch.py 2048 8192 200
```

---

## Reference

Milakov, M., & Gimelshein, N. (2018). *Online normalizer calculation
for softmax*. arXiv:1805.02867.
