"""Build each CUDA kernel, check it against torch.softmax, then time it.

Verification discipline matches the matmul work: many random elements, plus
a hand-computed 4-element row for the online merge, plus a large-logit
stress case that makes the unstable kernel overflow.
"""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ("naive", "stable", "online", "tiled")

def build():
    include = str(ROOT / "kernels")
    extra_cuda = [
        "-O3",
        "--use_fast_math",
        "-Xcompiler=/Zc:preprocessor",
    ]
    modules = {}
    for name in KERNELS:
        source = ROOT / "kernels" / f"softmax_{name}.cu"
        modules[name] = load(
            name=f"cuda_softmax_{name}",
            sources=[str(source)],
            extra_include_paths=[include],
            extra_cuda_cflags=extra_cuda,
            verbose=False,
        )
    return modules


def cuda_ms(fn, x, warmup=10, repeats=100):
    for _ in range(warmup):
        fn(x)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(repeats):
        fn(x)

    end.record()
    end.synchronize()

    return start.elapsed_time(end) / repeats


def max_abs_error(actual, reference):
    if not torch.isfinite(actual).all():
        n_bad = (~torch.isfinite(actual)).sum().item()
        return float("inf"), n_bad
    return torch.max((actual - reference).abs()).item(), 0


def check_many_elements(actual, reference, n_samples=4096):
    """Compare thousands of random locations, not a handful of entries."""
    flat_a = actual.reshape(-1)
    flat_r = reference.reshape(-1)
    n = flat_a.numel()
    idx = torch.randint(0, n, (min(n_samples, n),), device=actual.device)
    torch.testing.assert_close(flat_a[idx], flat_r[idx], rtol=1e-5, atol=1e-6)


def hand_example_online(module):
    """Row [1, 2, 3, 0]: expected softmax can be computed by hand."""
    x = torch.tensor([[1.0, 2.0, 3.0, 0.0]], device="cuda")
    # max = 3, sum = e^{-2} + e^{-1} + 1 + e^{-3}
    m = 3.0
    d = sum(torch.exp(torch.tensor(v - m)) for v in (1.0, 2.0, 3.0, 0.0))
    expected = torch.exp(x - m) / d
    actual = module.softmax(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA-enabled PyTorch install is required.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"SMs: {props.multi_processor_count}, mem: {props.total_memory / 1e9:.1f} GB")
    print()

    modules = build()

    # --- correctness: moderate random values (all kernels, including naive) ---
    x_ok = torch.randn(512, 1024, device=device) * 2.0
    ref_ok = torch.softmax(x_ok, dim=1)
    print("Correctness on N(0, 4) inputs (512 x 1024), 4096 random samples:")
    for name, module in modules.items():
        actual = module.softmax(x_ok)
        check_many_elements(actual, ref_ok)
        err, _ = max_abs_error(actual, ref_ok)
        print(f"  {name:<8} max |err| = {err:.3e}  OK")
    print()

    hand_example_online(modules["online"])
    hand_example_online(modules["tiled"])
    print("Hand example [1, 2, 3, 0]: online and tiled match closed form. OK")
    print()

    # --- overflow stress: values in the hundreds ---
    x_big = torch.randn(256, 2048, device=device) * 8.0 + 100.0
    ref_big = torch.softmax(x_big, dim=1)
    naive_big = modules["naive"].softmax(x_big)
    naive_finite = torch.isfinite(naive_big).all().item()
    print("Overflow stress (values ~ N(100, 64), 256 x 2048):")
    print(f"  torch.softmax finite: {torch.isfinite(ref_big).all().item()}")
    print(f"  naive finite:         {naive_finite}  "
          f"(False is expected — expf overflows above ~88.7)")
    for name in ("stable", "online", "tiled"):
        actual = modules[name].softmax(x_big)
        check_many_elements(actual, ref_big)
        err, _ = max_abs_error(actual, ref_big)
        print(f"  {name:<8} finite, max |err| = {err:.3e}")
    print()

    # --- timing ---
    rows, cols = 1024, 4096
    x = torch.randn(rows, cols, device=device)
    bytes_elem = x.numel() * x.element_size()
    # Theoretical traffic (paper §2–3): naive 3, safe 4, online 3, tiled 2.
    traffic = {
        "naive": 3 * bytes_elem,
        "stable": 4 * bytes_elem,
        "online": 3 * bytes_elem,
        "tiled": 2 * bytes_elem,
        "torch": 3 * bytes_elem,  # typical fused safe+write, conservative
    }

    print(f"Benchmark: {rows} x {cols} float32, 10 warmup + 100 timed runs")
    print(f"{'kernel':<10} {'ms':>8} {'GB/s (model)':>14} {'max |err|':>12}")
    times = {}
    for name, module in modules.items():
        actual = module.softmax(x)
        ref = torch.softmax(x, dim=1)
        err, n_bad = max_abs_error(actual, ref)
        ms = cuda_ms(module.softmax, x)
        times[name] = ms
        gbps = traffic[name] / (ms * 1e-3) / 1e9
        err_s = "inf/nan" if n_bad else f"{err:.3e}"
        print(f"{name:<10} {ms:8.3f} {gbps:14.1f} {err_s:>12}")

    torch_ms = cuda_ms(lambda t: torch.softmax(t, dim=1), x)
    print(f"{'torch':<10} {torch_ms:8.3f} {traffic['torch'] / (torch_ms * 1e-3) / 1e9:14.1f} {'reference':>12}")
    print()
    print("Bandwidth is modeled from the paper's access counts")
    print("(safe=4, online/naive=3, tiled=2), not Nsight.")


if __name__ == "__main__":
    main()
