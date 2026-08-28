# CUDA softmax — notes

Paper: Milakov & Gimelshein, *Online normalizer calculation for softmax*,
arXiv:1805.02867 (2018). Short NVIDIA note: the classical softmax, not a
new probability function.

---

## Pass 1 (skim)

**Problem.** Softmax on a length-V vector is memory-bandwidth bound. The
*naive* form does two scans (sum of exp, then normalize) — 3 memory accesses
per element — and overflows float32. The *safe* form used by every major
framework subtracts `max(x)` first, which needs a third scan — 4 accesses
per element.

**Core idea.** Keep a running max `m` and a running sum `d = Σ exp(x_i - m)`
together. When a new value raises the max, rescale the old sum by
`exp(old_m - new_m)` before adding the new term. One pass for the normalizer
instead of two. Output still needs a second pass over x, so total traffic
drops from 4 to 3.

**What they showed.** On V100, online softmax is up to ~1.3× faster than safe
softmax once V is large enough to miss cache (~4k). Fusing with TopK (beam
search) can hit ~5× because TopK does not need every y_i written. The operator
they define on `(m, d)` pairs is associative, so it parallelizes.

---

## Pass 2 — recurrence (the thing to implement)

Safe softmax:
y_i = exp(x_i - m_V) / sum_j exp(x_j - m_V), where m_V = max_k x_k

Online update for each new x_j (Algorithm 3):
Initialize m_0 = -∞ and d_0 = 0.

For each new value x_j, update the running maximum as:
m_j = max(m_{j-1}, x_j)

Then update the running denominator as:
d_j = d_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)

After processing all V elements, the output is:
y_i = exp(x_i - m_V) / d_V

**Why the rescale is correct.** After j-1 steps,
d_{j-1} = sum_{i<j} exp(x_i - m_{j-1}).

If the maximum does not change, then exp(m_{j-1} - m_j) = 1, so you simply
add the new term exp(x_j - m_j).

If x_j becomes the new maximum, every previous term needs to be expressed
relative to the new maximum. For each previous element:

exp(x_i - x_j) = exp(x_i - m_{j-1}) * exp(m_{j-1} - x_j)

Therefore, multiplying the entire old sum by exp(m_{j-1} - m_j) converts
the old normalization base to the new one. Theorem 1 in the paper formalizes
this by induction.

**Parallel merge (eq. 4).** Two partial results, (m_a, d_a) and (m_b, d_b),
can be merged using the same rescaling rule:

m = max(m_a, m_b)
d = d_a * exp(m_a - m) + d_b * exp(m_b - m)

This is the same operation as processing one new element, where an individual
element can be viewed as the pair (x, 1). Because this merge operation is
associative and commutative, partial results can be combined using a tree
reduction across CUDA threads. The identity element is (-∞, 0).

**Vocabulary I looked up rather than guessing**

- *Normalizer / d_V*: the denominator sum exp(x_j - m), not a separate
  "layer norm".
- *Safe softmax*: max-subtracted form (2), not "checked for NaNs".
- *Online algorithm*: incremental statistics in one pass (Welford's variance
  is the cited analogy), not "online learning".
- *Associative reduction*: order of merging pairs does not change the
  mathematical result (float rounding still can).
- *Warp shuffle* (`__shfl_down_sync`): register-to-register exchange inside
  32 threads; no shared memory for the intra-warp step.

**Hand example** used to debug merge before any large tensor: row `[1, 2, 3, 0]`.

| j | x | m | d |
|---|---|---|---|
| 1 | 1 | 1 | 1 |
| 2 | 2 | 2 | exp(-1) + 1 |
| 3 | 3 | 3 | exp(-2) + exp(-1) + 1 |
| 4 | 0 | 3 | exp(-2) + exp(-1) + 1 + exp(-3) |

Softmax = `[exp(-2), exp(-1), 1, exp(-3)] / d`. The benchmark asserts this
closed form for `softmax_online` and `softmax_tiled`.

---

## Pass 3 — implementation mapping

- One **block per row**, 256 threads, stride over columns.
- **Naive (`softmax_naive.cu`)**: Algorithm 1 only. `exp(x)` then divide.
  Overflows on logits ≳ 89.
- **Stable**: three global passes + shared-memory max reduction then sum
  reduction. Identity for unused threads: `-FLT_MAX` for max, `0` for sum.
- **Online**: each thread's loop is Algorithm 3; the block reduction is eq. 4
  on `SoftmaxPair`. Output still rereads global memory.
- **Tiled**: load the row once into smem; warp-shuffle `merge_pair` via
  `__shfl_down_sync`; 8 warp results in a tiny smem array; warp 0 finishes;
  write from the tile. Rows that do not fit in 48 KiB (~12k floats) are
  rejected.

---

## Why max-subtraction does not change the math

exp(x_i) / sum_j exp(x_j)
= [exp(x_i) * exp(-m)] / [sum_j exp(x_j) * exp(-m)]
= exp(x_i - m) / sum_j exp(x_j - m)

Largest exponent is exp(0) = 1. `expf` of a float32 overflows near 88.72
(`log(FLT_MAX)`). Logits in the hundreds are a realistic classifier head;
the unstable kernel is there to *show* that failure, not as a candidate.

---

## Bugs that actually matter (reduction, same class as matmul tiling edges)

**1. Max identity is `-inf`, not `0`.** With 256 threads and a 4-element row,
252 threads never touch data. If those threads enter the max-reduction with
`local_max = 0`, the block max becomes `max(true_max, 0)`. Softmax is
invariant to subtracting a *slightly* larger-than-true max, so `[-1, -2]`
still looks fine. The failure is a *much* larger fake max: row `[-80, -90]`,
reported max `0`, then `exp(-80)` underflows to 0 and you get `0/0 → nan`.
`-FLT_MAX` is the identity of `fmaxf` and does not invent a fake origin.
(Initializing to `+FLT_MAX` is the other disaster: every `exp(x - inf)` = 0.)

**2. Adding partial sums without aligning maxes (the online bug).** Thread A
saw `[0, 1]` → (m=1, d=exp(-1)+1). Thread B saw `[3]` → (m=3, d=1). Summing
d's gives ~1.37 + 1, but the correct merge is
`d = (exp(-1)+1)*exp(1-3) + 1 = exp(-2)+exp(-1)+1`. Off by a factor. This is
the analogue of the integer-division tile-boundary miss in matmul: the code
looks like a reduction, the types match, and the output even sums to 1 — so
a weak "check a few elements" test can miss it. The hand row `[1,2,3,0]` with
256 threads (most idle) forces a real cross-thread merge. That is why the
closed-form test exists.

**3. Shared-memory alignment after an odd-length tile.** `float tile[cols]`
followed by `SoftmaxPair*` is misaligned when `cols` is odd. Misaligned
float2-sized struct is undefined on some devices. Pad tile bytes to a
multiple of 8.

**4. Warp 0 must feed identity pairs into unused shuffle lanes.** With 256
threads there are 8 warps. Warp 0's `__shfl_down_sync` still runs on all 32
lanes. Lanes 8–31 must start as `empty_pair()`, not `warp_partials[lane]`
(out of bounds) or `{0,0}` (max 0 contaminates a negative row).

---

## Benchmark log — softmax kernels

Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU (20 SMs, 6.4 GB). Windows,
display-active GPU — same variance caveat as the matmul tables: wall times
jitter a few percent run-to-run; do not over-interpret 0.01 ms gaps.
Command: `python benchmarks/benchmark_pytorch.py`. Nine consecutive runs,
then means.

Modeled DRAM traffic (paper): naive 3N, stable 4N, online 3N, tiled 2N.
Softmax is bandwidth-bound; GB/s here is *modeled* from those counts, not
Nsight. Modeled GB/s is not comparable across kernels (stable's 4-pass model
inflates GB/s even when it is slower). Wall time is the primary metric.

**Correctness (identical on every run).** N(0, 4) 512×1024, 4096 samples:
naive/stable/online max |err| = 1.192e-07; tiled = 1.788e-07. Overflow
stress N(100, 64) 256×2048: torch/stable/online/tiled finite; naive not
(expected). Stress max |err|: stable 2.384e-07, online 1.788e-07,
tiled 3.576e-07. Timing-shape errors vs torch: ~1.9e-09–3.7e-09. All of
that is ordinary float32 rounding, not a bug. Hand row `[1, 2, 3, 0]`:
online and tiled match closed form.

**Mean of 10 runs, 1024×4096 float32, 10 warmup + 100 timed launches**

| Kernel | Time (ms) | Range | Modeled GB/s |
|---|---|---|---|
| Naive | 0.333 | 0.327–0.340 | 151.0 |
| Stable | 0.403 | 0.393–0.416 | 166.7 |
| Online | 0.318 | 0.310–0.337 | 158.5 |
| Tiled | 0.225 | 0.220–0.232 | 149.0 |
| torch.softmax | 0.224 | 0.220–0.229 | 224.8 |

Ranking is stable across all 10 runs: stable slowest, then naive, then
online, then tiled ≈ torch.

**What the numbers actually say**

- Online is ~1.27× faster than stable on the same "safe" math. That is the
  paper's claim on this GPU: dropping the extra global pass is worth more
  than the rescale arithmetic.
- Tiled is ~1.41× faster than online (and matches torch). The row stays in
  shared memory for the write, so traffic is 2N instead of 3N.
- Naive is *faster* than stable (~0.333 vs ~0.403 ms) even though both scan
  the row. Naive does fewer `expf` calls (no max-subtract / rescale) and less
  shared-memory reduction work. It is arithmetically lighter and numerically
  wrong on large logits.
- Stable's modeled GB/s is *higher* than online's while wall time is worse.
  The model charges stable 4 passes and online 3, so the same millisecond
  would look 33% "more bandwidth" for stable. The extra pass costs more than
  proportionally — overhead beyond the access-count sketch.

Nsight Systems: not captured. If added later, use kernel DRAM throughput,
not the Python wrapper.

---

## Benchmark log — fused matmul + softmax

**What the fused kernel does.** One block per query row. Pass 1 streams K in
THREADS×TILE_D panels into shared memory, computes dot products with the
cached Q row, and merges each score into a running SoftmaxPair using
merge_pair. Pass 2 recomputes the same dot products and writes
P = softmax(S) directly. S = QK^T is never written to HBM.

**Correctness.** Verified at three levels:
- Random N(0,1) Q,K (32×48, D=64), 4096 sampled elements vs
  torch.softmax(Q @ K.T): max |err| = 1.103e-06. OK.
- Non-multiple-of-tile edge sizes (17×19, D=13): OK.
- Closed-form hand example where Q @ K^T = [1,2,3,0] by construction: OK.
- Overflow stress (scores in the hundreds): fused finite,
  max |err| = 5.96e-08. OK.

**Mean of 5 sessions, 10 warmup + 100 timed runs**

| Kernel | M=N=1024, D=64 (ms) | M=N=4096, D=64 (ms) |
|---|---|---|
| Fused (no S in HBM) | 4.256 | 66.394 |
| Unfused tiled (our softmax) | 0.111 | 1.424 |
| Unfused torch | 0.108 | 1.444 |

**Why the fused kernel is slower — the synchronization bottleneck.**

The naive two-pass fusion exposes a fundamental bottleneck unrelated to
memory access. `dot_k_tile` issues two `__syncthreads()` calls per K-tile
panel, called in a loop over N/THREADS panels, twice per kernel (pass 1 and
pass 2). At N=4096, THREADS=256, this accumulates 16 panels × 2 passes × 2
sync calls = 64 block-wide synchronizations per output row. Each
`__syncthreads()` stalls the entire block until every warp reaches that
point, serializing execution across the loop.

The modeled bandwidth of 1.1 GB/s at both problem sizes confirms this
diagnosis: the GPU's measured bandwidth on the tiled softmax is ~192 GB/s,
so 1.1 GB/s means the kernel is almost entirely idle — waiting at
synchronization barriers rather than moving data or computing.

At M=N=1024, the intermediate S matrix is 4MB and likely fits in L2 cache
(~24MB on RTX 3050), so the unfused path does not pay real HBM cost for S
anyway. At M=N=4096, S = 64MB — genuinely exceeding L2 — but the
synchronization overhead in the fused kernel is so large that eliminating
S from HBM provides no net benefit.

**What Flash Attention does differently.**

Flash Attention (Dao et al., 2022) solves the same problem with a different
execution model: the outer loop iterates over K/V tiles rather than query
rows, each block accumulates partial softmax statistics across tiles without
needing per-tile block-wide synchronization, and tile sizes are chosen so
that one GEMM tile plus online statistics fit in shared memory simultaneously.
This eliminates the per-tile `__syncthreads()` overhead entirely. The naive
two-pass approach here has the right mathematical idea — online statistics,
no HBM for S — but the wrong execution structure for achieving a performance
win.

The natural next step is restructuring the outer loop as Flash Attention
describes, and fusing P @ V so that P is also never materialized.

---

## Write-up fragments (README / report / post)

**One sentence each**

- Stability: subtract max so `exp` never sees a positive argument.
- Online: keep `(m, d)` so the max and the sum share one load of x.
- Tiling: keep that load in shared memory and finish the reduction with
  shuffles so the write pass does not touch DRAM again.
- Fused: compute QK^T scores on the fly while building online statistics so
  S never materialises, then recompute scores for the output write.

**Future work (report closing line).** Restructure the fused kernel with
Flash Attention's execution model — outer loop over K/V tiles, no per-tile
block synchronization, fused P@V — so that neither S nor P is materialized.

**Blog title.** "Why naive softmax overflows and how to fix it in CUDA."
Story to tell: the unstable kernel going inf on logits of 100; the false
confidence of "outputs sum to 1" when merge forgot to rescale; link the
matmul post; name FlashAttention as the reason both kernels belong in one
portfolio; end with the fused kernel result and the synchronization diagnosis
as a preview of what Flash Attention had to solve.
