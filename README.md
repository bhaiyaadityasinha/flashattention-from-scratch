# CUDA Softmax

A from-scratch CUDA softmax project implementing four variants — unstable,
stable, online, and tiled — and a fused matmul+softmax kernel that eliminates
the intermediate attention score matrix from HBM.

The project implements the online normalizer algorithm from Milakov &
Gimelshein (2018), demonstrates its core claim empirically (eliminating one
global memory pass reduces wall time by ~27%), and extends it toward the
fused attention computation at the heart of Flash Attention.

---

## Benchmark results — softmax kernels

**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere, sm\_86,
20 SMs, 6.4 GB) · 1024×4096 FP32
**Software:** CUDA 13.0 · PyTorch 2.13.0 · `-O3 --use_fast_math`
**Methodology:** mean of 9 consecutive sessions (10 warmup + 100 timed
runs each) · each kernel built via `torch.utils.cpp_extension.load`

| Kernel | Method | Time (ms) | Range (ms) | Modeled GB/s |
|---|---|---|---|---|
| Naive | Unstable `exp(x)`, no max subtraction | 0.333 | 0.327–0.340 | 151.0 |
| Stable | Three global passes, max subtraction | 0.403 | 0.393–0.416 | 166.7 |
| Online | Running `(max, sum)` pairs, one stats pass | 0.318 | 0.310–0.337 | 158.5 |
| Tiled | Shared-memory row + `__shfl_down_sync` | 0.225 | 0.220–0.232 | 149.0 |
| torch.softmax | Vendor kernel (reference) | 0.224 | 0.220–0.229 | 224.8 |

*Modeled bandwidth is computed from the paper's theoretical access counts
(naive/online = 3N, stable = 4N, tiled = 2N bytes) and is not directly
comparable across kernels — PyTorch likely uses async memory operations
internally. Wall time is the primary metric.*

**Key findings:**

- Online is 27% faster than stable (0.318 ms vs 0.403 ms), directly
  confirming that eliminating one global memory pass reduces wall time on
  real hardware — the central claim of Milakov & Gimelshein.
- Tiled (0.225 ms) matches `torch.softmax` (0.224 ms) within measurement
  noise across all nine sessions.
- Naive is faster than stable (0.333 ms vs 0.403 ms) despite being
  numerically incorrect. Without max subtraction, fewer `expf` calls are
  needed and shared memory operations are simpler — arithmetically lighter,
  numerically broken on large inputs.
- Stable's modeled GB/s (166.7) exceeds online's (158.5) while its wall
  time is worse. The model charges stable 4 passes vs online's 3, so the
  same time would look 33% "more bandwidth" for stable. The extra pass
  costs more than proportionally — overhead beyond the access-count sketch.

---

## Benchmark results — fused matmul + softmax

**What the fused kernel does.** One block per query row. Pass 1 streams K
in THREADS×TILE\_D panels, computes dot products with the cached Q row, and
merges each score into a running `SoftmaxPair` using `merge_pair`. Pass 2
recomputes the same dot products and writes P = softmax(S) directly.
S = QK^T is never written to HBM.

**Correctness.** Verified at three levels: random N(0,1) Q,K (32×48, D=64),
4096 sampled elements vs `torch.softmax(Q @ K.T)` (max |err| = 1.103e-06);
non-multiple-of-tile edge sizes (17×19, D=13); closed-form hand example
where Q @ K^T = [1, 2, 3, 0] by construction. Overflow stress with scores
in the hundreds: fused finite, max |err| = 5.96e-08.

**Methodology:** mean of 4 consecutive sessions, 10 warmup + 100 timed runs.

| Kernel | M=N=1024, D=64 (ms) | M=N=4096, D=64 (ms) |
|---|---|---|
| Fused (no S in HBM) | 4.256 | 66.394 |
| Unfused tiled (our softmax) | 0.111 | 1.424 |
| Unfused torch | 0.108 | 1.444 |

*Modeled bandwidth: fused ~1.1 GB/s at both sizes, confirming the
bottleneck is synchronization, not memory access.*

**Why the fused kernel is slower — and what Flash Attention does instead.**

The naive two-pass fusion exposes a synchronization bottleneck.
`dot_k_tile` issues two `__syncthreads()` calls per K-tile panel, called
in a loop over N/THREADS panels, twice per kernel. At N=4096, THREADS=256,
this accumulates 64 block-wide synchronizations per output row, stalling
the entire block at each one. The 1.1 GB/s modeled bandwidth — against the
GPU's ~192 GB/s measured bandwidth on the tiled softmax — confirms the
kernel is almost entirely idle at synchronization barriers rather than
moving data or computing.

At M=N=1024, S = 4 MB fits in L2 cache (~24 MB on RTX 3050), so the
unfused path does not pay real HBM cost for S. At M=N=4096, S = 64 MB
genuinely exceeds L2, but the synchronization overhead dominates regardless.

Flash Attention (Dao et al., 2022) addresses this with a different execution
model: the outer loop iterates over K/V tiles rather than query rows, each
block accumulates partial softmax statistics across tiles without per-tile
block-wide synchronization, and tile sizes are chosen so that one GEMM tile
plus online statistics fit in shared memory simultaneously. The naive
two-pass approach here has the right mathematical insight — online statistics,
no HBM for S — but the wrong execution structure for a performance win.

The natural next step is restructuring the outer loop as Flash Attention
describes and fusing P @ V so that neither S nor P is materialised.

---

## Kernel variants

**Naive (`softmax_naive.cu`)**
No max subtraction. Computes `exp(x)` directly and divides by the sum.
`expf` overflows float32 for inputs above ~88.7, producing `inf` or `nan`
for rows with large values. Included as a deliberate control case that
demonstrates the overflow failure mode — verified to produce non-finite
outputs on inputs drawn from N(100, 64).

**Stable (`softmax_stable.cu`)**
Three global memory passes: (1) row max, (2) sum of `exp(x − max)`,
(3) write `exp(x − max) / sum`. Subtracting the row max guarantees all
exponents are ≤ 0, so nothing overflows. The output pass re-reads the row
from global memory a third time — the pass that online softmax eliminates.

**Online (`softmax_online.cu`)**
Single-pass statistics using the Milakov & Gimelshein recurrence (Algorithm
3 + parallel merge, eq. 4). Each thread maintains a running `(max, sum)`
pair; when a new element exceeds the current max, the running sum is
rescaled by `exp(old_max − new_max)`. Partial pairs merged across the block
with the same rule. Eliminates one global memory pass vs stable. Output
still reads the row a second time.

**Tiled (`softmax_tiled.cu`)**
Cooperatively loads the entire row into shared memory once. Statistics
computed via `warp_reduce_pair` (`__shfl_down_sync` reduction of
`SoftmaxPair`); one warp merges per-warp results. Output written from the
shared-memory copy — no second global memory read. Reduces to 2 memory
passes (1 read + 1 write). Requires the row to fit in 48 KiB of shared
memory (checked at launch). Matches `torch.softmax` wall time on this
hardware.

**Fused matmul + softmax (`softmax_fused_matmul.cu`)**
Computes softmax(Q @ K^T) without writing S to HBM. Q row cached in shared
memory; K streamed in tiles. Two-pass design: pass 1 builds online
`(max, sum)` statistics across K tiles; pass 2 recomputes dot products and
writes P directly. Correct and numerically stable. Slower than unfused on
this GPU due to per-tile synchronization overhead — see benchmark section
above for full analysis.

---

## Correctness

Each softmax kernel is verified at three levels:

1. **Random inputs** — 4096 randomly sampled elements checked against
   `torch.softmax` on 512×1024 inputs drawn from N(0, 4). All errors
   within 1e-7 to 4e-7 (normal single-precision rounding).

2. **Hand example** — `[1, 2, 3, 0]` checked against a closed-form
   expected value for online and tiled, independent of PyTorch.

3. **Overflow stress** — 256×2048 inputs drawn from N(100, 64). Naive
   produces non-finite outputs (verified). Stable, online, and tiled
   all produce finite results matching `torch.softmax`.

The fused kernel is verified against `torch.softmax(Q @ K.T)` using the
same three-level discipline plus a non-multiple-of-tile edge case.

---

## Repository layout

```
kernels/
    common.cuh                      SoftmaxPair, merge_pair, warp_reduce_pair
    softmax_naive.cu                Unstable baseline (overflow control case)
    softmax_stable.cu               Three-pass numerically stable softmax
    softmax_online.cu               Online single-pass softmax (Milakov & Gimelshein)
    softmax_tiled.cu                Tiled + warp-shuffle softmax
    softmax_fused_matmul.cu         Fused softmax(Q @ K^T) — S never written to HBM
benchmarks/
    benchmark_pytorch.py            Build, verify, and time all kernels vs PyTorch
requirements.txt
notes.md                            Development log — bugs, decisions, benchmark history
```

---

## Build and run

Requires PyTorch with CUDA support. Kernels are compiled via
`torch.utils.cpp_extension.load` at runtime — no manual `nvcc` invocation
needed.

```bash
pip install -r requirements.txt
python benchmarks/benchmark_pytorch.py
```

Optional arguments (rows, cols, iterations) for the softmax benchmark:

```bash
python benchmarks/benchmark_pytorch.py 2048 8192 200
```

---

## Reference

Milakov, M., & Gimelshein, N. (2018). *Online normalizer calculation for
softmax*. arXiv:1805.02867.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention:
Fast and memory-efficient exact attention with IO-awareness*. NeurIPS 2022.
