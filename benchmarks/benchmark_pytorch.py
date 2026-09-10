from pathlib import Path
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ("naive", "stable", "online", "tiled")
FLAGS = ["-O3", "--use_fast_math", "-Xcompiler=/Zc:preprocessor"]


def load_kernel(name, source):
    return load(name=name, sources=[str(source)],
                extra_include_paths=[str(ROOT / "kernels")],
                extra_cuda_cflags=FLAGS, verbose=False)


def build():
    # Build each softmax kernel as a separate PyTorch extension.
    return {n: load_kernel(f"cuda_softmax_{n}",
            ROOT / "kernels" / f"softmax_{n}.cu") for n in KERNELS}


def build_fused():
    return load_kernel("cuda_softmax_fused_matmul",
                       ROOT / "kernels" / "softmax_fused_matmul.cu")


def build_flash():
    return load_kernel("cuda_flash_attention_fwd",
                       ROOT / "kernels" / "flash_attention_fwd.cu")


def cuda_ms(fn, *args, warmup=10, repeats=100):
    # Warm up first so initialization/JIT work is not included.
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(repeats):
        fn(*args)

    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def error(x, ref):
    if not torch.isfinite(x).all():
        return float("inf"), (~torch.isfinite(x)).sum().item()
    return (x - ref).abs().max().item(), 0


def check(x, ref, n=4096):
    # Sample large tensors for the correctness check.
    a, b = x.reshape(-1), ref.reshape(-1)
    i = torch.randint(0, a.numel(), (min(n, a.numel()),), device=x.device)
    torch.testing.assert_close(a[i], b[i], rtol=1e-5, atol=1e-6)


def hand():
    x = torch.tensor([[1., 2., 3., 0.]], device="cuda")
    m = x.max(-1, keepdim=True).values
    y = torch.exp(x - m)
    return x, y / y.sum(-1, keepdim=True)


def test_standalone(modules):
    x = torch.randn(512, 1024, device="cuda") * 4
    ref = torch.softmax(x, -1)

    print("Correctness: N(0, 4), 512 x 1024")
    for n, mod in modules.items():
        y = mod.softmax(x)
        check(y, ref)
        e, _ = error(y, ref)
        print(f"  {n:<7} max |err| = {e:.3e}  OK")

    # A small hand-check makes the expected values easy to verify.
    x, ref = hand()
    for n in ("online", "tiled"):
        torch.testing.assert_close(
            modules[n].softmax(x), ref, rtol=1e-6, atol=1e-6
        )
    print("Hand example [1, 2, 3, 0]: online/tiled OK")

    # Large positive inputs expose overflow in naive softmax.
    x = torch.randn(256, 2048, device="cuda") * 64 + 100
    ref = torch.softmax(x, -1)

    print("\nOverflow stress: N(100, 64), 256 x 2048")
    print(f"  torch.softmax finite: {torch.isfinite(ref).all().item()}")

    for n, mod in modules.items():
        y = mod.softmax(x)
        e, bad = error(y, ref)
        print(f"  {n:<7} " +
              ("finite: False" if bad else f"finite, max |err| = {e:.3e}"))


def benchmark(modules):
    M, N = 1024, 4096
    x = torch.randn(M, N, device="cuda")
    ref = torch.softmax(x, -1)

    # Approximate global-memory traffic for each implementation.
    traffic = {"naive": 3, "stable": 4, "online": 3, "tiled": 2}

    print(f"\nBenchmark: {M} x {N} float32, 10 warmup + 100 runs")
    print(f"{'kernel':<12} {'ms':>8} {'GB/s':>12} {'max |err|':>12}")

    for n, mod in modules.items():
        y = mod.softmax(x)
        e, _ = error(y, ref)
        ms = cuda_ms(mod.softmax, x)
        gb = M * N * 4 * traffic[n] / (ms * 1e-3) / 1e9
        print(f"{n:<12} {ms:8.3f} {gb:12.1f} {e:12.3e}")

    ms = cuda_ms(torch.softmax, x, -1)
    gb = M * N * 8 / (ms * 1e-3) / 1e9
    print(f"{'torch':<12} {ms:8.3f} {gb:12.1f} {'reference':>12}")
    print("\nBandwidth is modeled from theoretical access counts, not measured via Nsight.")


def fused_test(tiled):
    fused = build_fused()

    def ref(q, k):
        return torch.softmax(q @ k.T, -1)

    print("\n=== Fused matmul + softmax ===")
    print("S = QK^T is kept in tiles; scores are recomputed for P.")

    q = torch.randn(32, 64, device="cuda")
    k = torch.randn(48, 64, device="cuda")
    y, r = fused.fused_matmul_softmax(q, k), ref(q, k)
    check(y, r)
    e, _ = error(y, r)
    print(f"Correctness Q 32x64, K 48x64: {e:.3e}  OK")

    # Non-multiple dimensions test the boundary checks in the kernel.
    q = torch.randn(17, 13, device="cuda")
    k = torch.randn(19, 13, device="cuda")
    check(fused.fused_matmul_softmax(q, k), ref(q, k))

    q = torch.tensor([[1., 0.]], device="cuda")
    k = torch.tensor([[1., 0.], [2., 0.], [3., 0.], [0., 0.]],
                     device="cuda")
    torch.testing.assert_close(
        fused.fused_matmul_softmax(q, k), hand()[1],
        rtol=1e-6, atol=1e-6
    )
    print("Edge 17x19 D=13 + hand example: OK")

    # Large Q/K values produce large QK^T scores.
    q = torch.randn(16, 32, device="cuda") * 4 + 20
    k = torch.randn(64, 32, device="cuda") * 4 + 20
    y, r = fused.fused_matmul_softmax(q, k), ref(q, k)
    e, bad = error(y, r)
    print(f"Overflow stress: fused finite {torch.isfinite(y).all().item()}, "
          f"max |err| = {'inf/nan' if bad else f'{e:.3e}'}")

    print("\nFused benchmark:")
    print(f"{'Kernel':<18} {'Milliseconds':>12} "
          f"{'GB/s':>12} {'Max Error':>12}")
    print("-" * 58)

    for M, N, D in ((1024, 1024, 64), (4096, 4096, 64)):
        q = torch.randn(M, D, device="cuda")
        k = torch.randn(N, D, device="cuda")
        r = ref(q, k)

        bq, bk, bs = M * D * 4, N * D * 4, M * N * 4

        # Fused avoids writing the intermediate M x N score matrix.
        fused_bytes = 2 * (bq + bk) + bs
        unfused_bytes = bq + bk + 5 * bs

        def ours(a, b):
            return tiled.softmax(a @ b.T)

        def torch_path(a, b):
            return torch.softmax(a @ b.T, -1)

        y = fused.fused_matmul_softmax(q, k)
        e, bad = error(y, r)
        ms = cuda_ms(fused.fused_matmul_softmax, q, k)
        gb = fused_bytes / (ms * 1e-3) / 1e9
        print(f"\n{M}x{D} @ {N}x{D}")
        print(f"{'fused':<18} {ms:12.3f} {gb:12.1f} "
              f"{'inf/nan' if bad else f'{e:.3e}':>12}")

        y = ours(q, k)
        e, _ = error(y, r)
        ms = cuda_ms(ours, q, k)
        gb = unfused_bytes / (ms * 1e-3) / 1e9
        print(f"{'unfused tiled':<18} {ms:12.3f} {gb:12.1f} {e:12.3e}")

        ms = cuda_ms(torch_path, q, k)
        print(f"{'unfused torch':<18} {ms:12.3f} {'reference':>12}")

    print("\nFused-kernel GB/s figures are modeled from theoretical byte counts, not measured. They should not be used to infer the actual performance bottleneck without hardware profiling (e.g. Nsight Compute).")


def flash_attention_test():
    """Validate and time O = softmax(QK^T / sqrt(D)) V fairly.

    The naive fused baseline below deliberately materialises P because the
    original fused kernel ends at softmax(QK^T).  It is therefore a fair
    end-to-end attention baseline, not a comparison against softmax alone.
    """
    flash = build_flash()

    def ref(q, k, v, causal=False):
        return torch.nn.functional.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None], is_causal=causal
        )[0, 0]

    def naive_attention(q, k, v):
        # fused_matmul_softmax has no scaling argument, so scale Q first.
        p = fused.fused_matmul_softmax(q / (q.size(1) ** 0.5), k)
        return p @ v

    fused = build_fused()
    print("\n=== FlashAttention forward: O = softmax(QK^T / sqrt(D)) V ===")
    print("Neither S nor P is written by the FlashAttention kernel.")

    q = torch.randn(32, 64, device="cuda")
    k = torch.randn(48, 64, device="cuda")
    v = torch.randn(48, 64, device="cuda")
    y, r = flash.flash_attention_fwd(q, k, v), ref(q, k, v)
    check(y, r)
    e, _ = error(y, r)
    print(f"Random Q 32x64, K/V 48x64: {e:.3e}  OK")

    # Non-tile dimensions and a closed-form-friendly one-hot query exercise
    # bounds and the online recurrence independently of the random test.
    q = torch.tensor([[1., 0.]], device="cuda")
    k = torch.tensor([[1., 0.], [2., 0.], [3., 0.], [0., 0.]], device="cuda")
    v = torch.tensor([[1., 0.], [0., 1.], [2., 1.], [-1., 3.]], device="cuda")
    torch.testing.assert_close(flash.flash_attention_fwd(q, k, v), ref(q, k, v),
                               rtol=1e-6, atol=1e-6)
    q = torch.randn(17, 13, device="cuda")
    k = torch.randn(19, 13, device="cuda")
    v = torch.randn(19, 11, device="cuda")
    check(flash.flash_attention_fwd(q, k, v), ref(q, k, v))
    print("Hand example + non-multiple 17x19 (D=13, DV=11): OK")

    q = torch.randn(16, 32, device="cuda") * 4 + 20
    k = torch.randn(64, 32, device="cuda") * 4 + 20
    v = torch.randn(64, 32, device="cuda")
    y, r = flash.flash_attention_fwd(q, k, v), ref(q, k, v)
    e, bad = error(y, r)
    print(f"Overflow stress: flash finite {torch.isfinite(y).all().item()}, "
          f"max |err| = {'inf/nan' if bad else f'{e:.3e}'}")

    print("\nEnd-to-end attention benchmark (10 warmup + 100 runs):")
    print(f"{'Problem':<20} {'naive fused':>14} {'flash':>12} {'PyTorch SDPA':>15}")
    print("-" * 66)
    for M, N, D in ((1024, 1024, 64), (4096, 4096, 64)):
        q = torch.randn(M, D, device="cuda")
        k = torch.randn(N, D, device="cuda")
        v = torch.randn(N, D, device="cuda")
        r = ref(q, k, v)

        y = flash.flash_attention_fwd(q, k, v)
        check(y, r)
        e, _ = error(y, r)
        naive_ms = cuda_ms(naive_attention, q, k, v)
        flash_ms = cuda_ms(flash.flash_attention_fwd, q, k, v)
        sdpa_ms = cuda_ms(ref, q, k, v)
        print(f"{M}x{D} @ {N}x{D:<4} {naive_ms:14.3f} {flash_ms:12.3f} "
              f"{sdpa_ms:15.3f}  (flash err {e:.3e})")

    print("\nThe naive-fused column is fused softmax(QK^T) followed by P@V; it materialises P.")
    print("SDPA selects PyTorch's native backend. Compare wall time, but use Nsight Compute "
          "to attribute performance to synchronization, reuse, or memory stalls.")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SMs: {p.multi_processor_count}, "
          f"mem: {p.total_memory / 1e9:.1f} GB\n")

    modules = build()
    test_standalone(modules)
    benchmark(modules)
    fused_test(modules["tiled"])
    flash_attention_test()


if __name__ == "__main__":
    main()
