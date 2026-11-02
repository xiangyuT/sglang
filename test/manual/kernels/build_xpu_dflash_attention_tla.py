"""Build the opt-in B70 M4/M8 draft attention adapter from pinned local sources."""

import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

KERNEL_REV = "028ac5a4e92a4addabeedac1a94374515f68e4a5"
TLA_REV = "7b56b936b87f799f80dde60c2fd8d11c1128cebb"
HEADERS = (
    (
        "kernel/xe_fmha_fwd_kernel.hpp",
        "dflash_tla_kernel.hpp",
        "ddd420eef49757a14f3b057a3b9313cfd4b586e647e03a10df41941ffab848c9",
    ),
    (
        "xe_fmha_fwd_decode_runner.hpp",
        "dflash_tla_decode.hpp",
        "8f9b6f7fa7f90f07817006603f1400dba6a08cbf69285a21f9811021b8572d58",
    ),
)


def adapt_headers(kernel_source, destination, *, paged_kv=False, window_size=4):
    """Keep dependency checkouts unchanged and preserve their license notices."""
    root = kernel_source / "src/sycl/kernels/flash_attention_v2"
    records = []
    if window_size not in (4, 8) or (window_size == 8 and not paged_kv):
        raise ValueError("M8 requires the saved paged KV adapter")
    for relative, name, expected in HEADERS:
        original = root / relative
        raw = original.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Unexpected pinned attention header: {original}")
        text = before = raw.decode()
        identity = (
            "XeFMHAFwdDFlashPagedTlaKernel" if paged_kv else "XeFMHAFwdDFlashTlaKernel"
        )
        if window_size == 8:
            identity = "XeFMHAFwdDFlashPagedTlaM8Kernel"
        text = text.replace("XeFMHAFwdKernel", identity)
        if name == "dflash_tla_kernel.hpp":
            target = "auto full_tile_offset = seq_len_kv_cache - offset;"
            if text.count(target) != 1:
                raise ValueError("Cannot locate the pinned query-position calculation")
            text = text.replace(
                target,
                target
                + f"\n      // M{window_size} bidirectional draft: pseudo batch b is query b."
                + f"\n      full_tile_offset -= {window_size - 1} - (idx_b % {window_size});",
            )
        else:
            text = text.replace(
                '"sycl/kernels/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"',
                '"dflash_tla_kernel.hpp"',
            ).replace(
                "namespace decode",
                "namespace dflash_paged_tla_decode"
                if paged_kv
                else "namespace dflash_tla_decode",
            )
            if window_size == 8:
                text = text.replace(
                    "namespace dflash_paged_tla_decode",
                    "namespace dflash_paged_tla_decode_m8",
                )
        adapted = destination / name
        adapted.write_text(text)
        adapted.with_suffix(adapted.suffix + ".diff").write_text(
            "".join(
                difflib.unified_diff(
                    before.splitlines(True),
                    text.splitlines(True),
                    fromfile=relative,
                    tofile=name,
                )
            )
        )
        records.append(
            dict(
                original=str(original.resolve()),
                original_sha256=expected,
                adapted=str(adapted),
                adapted_sha256=hashlib.sha256(adapted.read_bytes()).hexdigest(),
            )
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--sycl-tla", type=Path, required=True)
    parser.add_argument("--compiler", default="icpx")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--paged-kv", action="store_true", help="Build the saved page128 KV adapter"
    )
    parser.add_argument("--window-size", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    if args.window_size == 8 and not args.paged_kv:
        parser.error("--window-size 8 requires --paged-kv")
    dependencies = []
    for path, revision in ((args.kernel_source, KERNEL_REV), (args.sycl_tla, TLA_REV)):
        git = ["git", "-c", "safe.directory=" + str(path.resolve()), "-C", str(path)]
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        if head != revision:
            parser.error(f"{path}: expected {revision}, found {head}")
        for extra in ([], ["--cached"]):
            subprocess.run([*git, "diff", "--quiet", *extra], check=True)
        dependencies.append(dict(path=str(path.resolve()), revision=head))
    output = args.output.resolve()
    receipt = output.with_suffix(output.suffix + ".json")
    headers = output.with_name(output.stem + "_headers")
    if any(p.exists() for p in (output, receipt, headers)):
        parser.error("Output, receipt or generated headers exist; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    headers.mkdir()
    adapted = adapt_headers(
        args.kernel_source,
        headers,
        paged_kv=args.paged_kv,
        window_size=args.window_size,
    )
    source_name = "attention_paged.cpp" if args.paged_kv else "attention.cpp"
    source = Path(__file__).resolve().parents[3] / (
        "python/sglang/kernels/jit/csrc/attention/xpu_dflash_tla/" + source_name
    )
    import torch

    torch_root = Path(torch.__file__).resolve().parent
    includes = [
        headers,
        torch_root / "include",
        torch_root / "include/torch/csrc/api/include",
        args.kernel_source / "src/sycl",
        args.kernel_source / "src",
        args.kernel_source / "include",
        args.sycl_tla / "include",
        args.sycl_tla / "tools/util/include",
    ]
    command = [
        args.compiler,
        "-std=c++20",
        "-fsycl",
        "-fPIC",
        "-shared",
        "-O3",
        "-fsycl-targets=spir64",
        "-Xspirv-translator",
        "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
        "-DCUTLASS_ENABLE_SYCL",
        "-DDFLASH_TLA_WINDOW=" + str(args.window_size),
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        *["-I" + str(p.resolve()) for p in includes],
        str(source),
        "-L" + str(torch_root / "lib"),
        "-Wl,-rpath," + str(torch_root / "lib"),
        "-ltorch_xpu",
        "-ltorch_cpu",
        "-lc10_xpu",
        "-lc10",
        "-o",
        str(output),
    ]
    metadata = dict(
        command=command,
        dependencies=dependencies,
        adapted_headers=adapted,
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        torch_version=torch.__version__,
        compiler=subprocess.check_output([args.compiler, "--version"], text=True),
        scope=("attention-paged-TLA" if args.paged_kv else "attention-TLA")
        + f" ABI1/B70/M{args.window_size}/BF16/H32/Hkv8/D128/window2047",
    )
    print(json.dumps(metadata, indent=2), flush=True)
    result = subprocess.run(command)
    metadata["returncode"] = result.returncode
    if result.returncode == 0:
        metadata["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(metadata, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
