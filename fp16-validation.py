#!/usr/bin/env python3
"""
FP16 validation harness for rusticl OpenCL targets.
Tests: extension presence, arithmetic correctness, MAD throughput,
       half<->float round-trip fidelity, vector widths (half2/half4/half8/half16).

Targets: gfx906 (Vega20 / Radeon Pro V340) via EPYC stack
         gfx1013 (Van Gogh / AMD BC-250) via Oberon

Usage:
    python3 validate_fp16_rusticl.py [--device-index N] [--all-devices] [--verbose]

Requirements:
    pip install pyopencl numpy --break-system-packages
"""

import sys
import argparse
import time
import numpy as np

try:
    import pyopencl as cl
except ImportError:
    sys.exit("pyopencl not found. Run: pip install pyopencl --break-system-packages")

KERNEL_SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void k_smoke(__global half *out) {
    half a = (half)1.5f;
    half b = (half)2.5f;
    out[0] = a + b;
    out[1] = a * b;
    out[2] = a - b;
    out[3] = (half)1.0f / b;
}

__kernel void k_roundtrip(__global const float *in_f,
                          __global float       *out_f,
                          int N) {
    int i = get_global_id(0);
    if (i >= N) return;
    half h  = (half)in_f[i];
    out_f[i] = (float)h;
}

__kernel void k_mad_half2(__global half2 *a, __global half2 *b,
                          __global half2 *c, __global half2 *out, int N) {
    int i = get_global_id(0);
    if (i >= N) return;
    out[i] = mad(a[i], b[i], c[i]);
}

__kernel void k_mad_half4(__global half4 *a, __global half4 *b,
                          __global half4 *c, __global half4 *out, int N) {
    int i = get_global_id(0);
    if (i >= N) return;
    out[i] = mad(a[i], b[i], c[i]);
}

__kernel void k_mad_half8(__global half8 *a, __global half8 *b,
                          __global half8 *c, __global half8 *out, int N) {
    int i = get_global_id(0);
    if (i >= N) return;
    out[i] = mad(a[i], b[i], c[i]);
}

__kernel void k_mad_half16(__global half16 *a, __global half16 *b,
                           __global half16 *c, __global half16 *out, int N) {
    int i = get_global_id(0);
    if (i >= N) return;
    out[i] = mad(a[i], b[i], c[i]);
}

__kernel void k_dot_half(__global const half *a,
                         __global const half *b,
                         __global float      *partial,
                         int N) {
    int i    = get_global_id(0);
    int lid  = get_local_id(0);
    int lsz  = get_local_size(0);
    __local float lmem[256];
    float acc = 0.0f;
    for (int j = i; j < N; j += get_global_size(0))
        acc += (float)a[j] * (float)b[j];
    lmem[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) lmem[lid] += lmem[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[get_group_id(0)] = lmem[0];
}

__kernel void k_throughput(__global half *out, int iters) {
    int i = get_global_id(0);
    half4 a = (half4)(1.1f, 0.9f, 1.05f, 0.95f);
    half4 b = (half4)(0.5f, 0.5f, 0.5f, 0.5f);
    half4 c = (half4)(0.1f, 0.1f, 0.1f, 0.1f);
    for (int k = 0; k < iters; k++) {
        a = mad(a, b, c);
        b = mad(b, a, c);
    }
    if (i == 0) out[0] = a.x + b.x;
}
"""

def check_ext(dev, ext_name="cl_khr_fp16"):
    return ext_name in dev.extensions.strip().split()

PASS_TAG = "[PASS]"
FAIL_TAG = "[FAIL]"
WARN_TAG = "[WARN]"
SKIP_TAG = "[SKIP]"

def tag(ok, warn=False):
    if ok is None:  return SKIP_TAG
    if warn:        return WARN_TAG
    return PASS_TAG if ok else FAIL_TAG


def validate_device(dev, verbose=False):
    name     = dev.name.strip()
    platform = dev.platform.name.strip()
    print("")
    print("=" * 70)
    print("  Device  : " + name)
    print("  Platform: " + platform)
    print("  Driver  : " + dev.driver_version)
    print("=" * 70)

    results = {}

    has_fp16 = check_ext(dev)
    results["cl_khr_fp16 extension"] = has_fp16
    print("  " + tag(has_fp16) + " cl_khr_fp16 extension present")
    if not has_fp16:
        print("        Extensions: " + dev.extensions[:120] + "...")
        print("  FP16 not supported on this device -- skipping kernel tests.")
        return results

    ctx = cl.Context([dev])
    q   = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    try:
        prg = cl.Program(ctx, KERNEL_SRC).build(options="-cl-std=CL2.0")
    except cl.RuntimeError as e:
        print("  " + FAIL_TAG + " Kernel build failed:\n" + str(e))
        results["kernel build"] = False
        return results
    results["kernel build"] = True
    print("  " + PASS_TAG + " Kernel build (cl_khr_fp16 enabled)")

    mf = cl.mem_flags

    # --- Smoke test ---
    out_h = np.zeros(4, dtype=np.float16)
    buf   = cl.Buffer(ctx, mf.WRITE_ONLY, size=out_h.nbytes)
    prg.k_smoke(q, (1,), None, buf)
    cl.enqueue_copy(q, out_h, buf)
    q.finish()
    expected  = np.array([4.0, 3.75, -1.0, 0.4], dtype=np.float16)
    smoke_ok  = np.allclose(out_h, expected, atol=0.01)
    results["smoke (add/mul/sub/div)"] = smoke_ok
    print("  " + tag(smoke_ok) +
          " Smoke: add/mul/sub/div  got=" + str(out_h) +
          "  exp=" + str(expected))

    # --- Round-trip ---
    test_vals = np.array([0.0, 1.0, -1.0, 0.5, 65504.0, 6e-5,
                          1.234, -3.14159, 0.001, 1000.0], dtype=np.float32)
    N = len(test_vals)
    buf_in  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=test_vals)
    buf_out = cl.Buffer(ctx, mf.WRITE_ONLY, size=test_vals.nbytes)
    prg.k_roundtrip(q, (N,), None, buf_in, buf_out, np.int32(N))
    out_rt = np.empty(N, dtype=np.float32)
    cl.enqueue_copy(q, out_rt, buf_out)
    q.finish()
    ref_rt = test_vals.astype(np.float16).astype(np.float32)
    rt_ok  = np.allclose(out_rt, ref_rt, rtol=1e-3, atol=1e-4)
    results["float->half->float round-trip"] = rt_ok
    if verbose or not rt_ok:
        for v, r, e in zip(test_vals, out_rt, ref_rt):
            ok_i = abs(r - e) < max(1e-4, 1e-3 * abs(e))
            print("        " + tag(ok_i) +
                  "  in=" + "{:12.6g}".format(v) +
                  "  gpu=" + "{:12.6g}".format(r) +
                  "  ref=" + "{:12.6g}".format(e))
    print("  " + tag(rt_ok) +
          " Half<->float round-trip (" + str(N) + " values)")

    # --- Vector MAD correctness ---
    for width, kname in [(2,  "k_mad_half2"),
                         (4,  "k_mad_half4"),
                         (8,  "k_mad_half8"),
                         (16, "k_mad_half16")]:
        Nv    = 1024
        dtype = np.float16
        rng   = np.random.default_rng(42)
        a_np  = rng.uniform(0.5, 1.5, Nv * width).astype(dtype)
        b_np  = rng.uniform(0.5, 1.5, Nv * width).astype(dtype)
        c_np  = rng.uniform(0.0, 0.5, Nv * width).astype(dtype)
        ref   = (a_np * b_np + c_np)

        ba = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=a_np)
        bb = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=b_np)
        bc = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=c_np)
        bo = cl.Buffer(ctx, mf.WRITE_ONLY, size=a_np.nbytes)
        kern = getattr(prg, kname)
        kern(q, (Nv,), None, ba, bb, bc, bo, np.int32(Nv))
        out_v = np.empty(Nv * width, dtype=np.float16)
        cl.enqueue_copy(q, out_v, bo)
        q.finish()
        max_err = float(np.max(np.abs(
            out_v.astype(np.float32) - ref.astype(np.float32))))
        vec_ok  = max_err < 1e-2
        label   = "half" + str(width) + " MAD correctness"
        results[label] = vec_ok
        print("  " + tag(vec_ok) +
              " half" + str(width) + " MAD  max_err=" +
              "{:.5f}".format(max_err))

    # --- Dot product reduction ---
    Nd      = 16384
    rng2    = np.random.default_rng(7)
    a_d     = rng2.uniform(-1, 1, Nd).astype(np.float16)
    b_d     = rng2.uniform(-1, 1, Nd).astype(np.float16)
    ref_dot = float(np.sum(a_d.astype(np.float64) * b_d.astype(np.float64)))
    lsz     = 256
    ngrp    = (Nd + lsz - 1) // lsz
    ba2     = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=a_d)
    bb2     = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=b_d)
    bpart   = cl.Buffer(ctx, mf.WRITE_ONLY, size=ngrp * 4)
    prg.k_dot_half(q, (ngrp * lsz,), (lsz,), ba2, bb2, bpart, np.int32(Nd))
    partial = np.empty(ngrp, dtype=np.float32)
    cl.enqueue_copy(q, partial, bpart)
    q.finish()
    gpu_dot = float(np.sum(partial))
    dot_err = abs(gpu_dot - ref_dot) / (abs(ref_dot) + 1e-9)
    dot_ok  = dot_err < 0.05
    results["half dot product reduction"] = dot_ok
    print("  " + tag(dot_ok) +
          " Half dot product  gpu=" + "{:.4f}".format(gpu_dot) +
          "  ref=" + "{:.4f}".format(ref_dot) +
          "  rel_err=" + "{:.4f}".format(dot_err))

    # --- Throughput benchmark ---
    print("")
    print("  --- Throughput benchmark (half4 MAD chain) ---")
    BENCH_ITERS = 2000
    BENCH_GSIZE = dev.max_compute_units * 256
    buf_tp = cl.Buffer(ctx, mf.WRITE_ONLY, size=2)
    prg.k_throughput(q, (BENCH_GSIZE,), None, buf_tp, np.int32(BENCH_ITERS))
    q.finish()
    t0 = time.perf_counter()
    prg.k_throughput(q, (BENCH_GSIZE,), None, buf_tp, np.int32(BENCH_ITERS))
    q.finish()
    t1      = time.perf_counter()
    elapsed = t1 - t0
    ops     = BENCH_GSIZE * BENCH_ITERS * 2 * 4 * 2
    gflops  = ops / elapsed / 1e9
    results["throughput_gflops"] = gflops
    print("  " + PASS_TAG +
          " half4 MAD throughput: " + "{:.1f}".format(gflops) +
          " GFLOPS  (elapsed " + "{:.1f}".format(elapsed * 1000) + " ms)")

    # --- Summary ---
    print("")
    print("  --- Summary ---")
    all_bool = {k: v for k, v in results.items() if isinstance(v, bool)}
    passed   = sum(all_bool.values())
    total    = len(all_bool)
    print("  Passed " + str(passed) + "/" + str(total) + " functional checks")
    for k, v in all_bool.items():
        print("    " + tag(v) + " " + k)
    tput = results.get("throughput_gflops")
    if isinstance(tput, float):
        print("    [----] throughput: " + "{:.1f}".format(tput) + " GFLOPS")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="FP16 rusticl validation harness (ASCII output)")
    parser.add_argument("--device-index", type=int, default=None,
                        help="Run on a specific global device index (0-based)")
    parser.add_argument("--all-devices", action="store_true",
                        help="Run on every available OpenCL device")
    parser.add_argument("--platform-filter", type=str, default=None,
                        help="Filter by platform/device name substring (case-insensitive)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-value round-trip table")
    args = parser.parse_args()

    all_devs = []
    for plat in cl.get_platforms():
        for dev in plat.get_devices():
            all_devs.append((plat, dev))

    print("Available OpenCL devices:")
    for idx, (plat, dev) in enumerate(all_devs):
        fp16_str = "fp16:YES" if check_ext(dev) else "fp16: NO"
        print("  [" + str(idx) + "] " +
              plat.name.strip() + " / " +
              dev.name.strip() + "  (" + fp16_str + ")")

    if args.platform_filter:
        f = args.platform_filter.lower()
        all_devs = [(p, d) for p, d in all_devs
                    if f in p.name.lower() or f in d.name.lower()]
        if not all_devs:
            sys.exit("No devices match filter '" + args.platform_filter + "'")

    if args.device_index is not None:
        target_devs = [all_devs[args.device_index]]
    elif args.all_devices:
        target_devs = all_devs
    else:
        fp16_devs   = [(p, d) for p, d in all_devs if check_ext(d)]
        target_devs = fp16_devs if fp16_devs else [all_devs[0]]
        if not fp16_devs:
            print("No fp16-capable devices found -- running smoke test on first device.")

    all_results = {}
    for plat, dev in target_devs:
        r = validate_device(dev, verbose=args.verbose)
        all_results[dev.name.strip()] = r

    if len(all_results) > 1:
        print("")
        print("=" * 70)
        print("  Cross-device comparison")
        print("=" * 70)
        keys = sorted({k for r in all_results.values()
                       for k in r if isinstance(r[k], bool)})
        dev_names = list(all_results.keys())
        header = "  " + "{:<45}".format("Test") + \
                 "".join("  " + "{:<18}".format(n[:18]) for n in dev_names)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for k in keys:
            row = "  " + "{:<45}".format(k)
            for r in all_results.values():
                v   = r.get(k)
                row += "  " + "{:<18}".format(tag(v))
            print(row)
        trow = "  " + "{:<45}".format("throughput_gflops")
        for r in all_results.values():
            v     = r.get("throughput_gflops")
            label = ("{:.1f} GFLOPS".format(v) if v else "N/A")
            trow += "  " + "{:<18}".format(label)
        print(trow)

    print("")
    print("Done.")
    print("")


if __name__ == "__main__":
    main()
