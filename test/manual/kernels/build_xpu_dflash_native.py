"""Build the opt-in B70 native DFlash expert/attention library from pinned local dependencies.

This does not install a wheel, change versions, or download dependencies.
See python/sglang/kernels/jit/csrc/moe/xpu_compact/README.md.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

KERNEL_REV = "028ac5a4e92a4addabeedac1a94374515f68e4a5"
TLA_REV = "7b56b936b87f799f80dde60c2fd8d11c1128cebb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--sycl-tla", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="icpx")
    parser.add_argument("--window-size", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    for path, revision in (
        (args.kernel_source, KERNEL_REV),
        (args.sycl_tla, TLA_REV),
    ):
        # These dependency directories were explicitly selected by the caller.
        # Permit their container bind-mount ownership for this command only.
        git = ["git", "-c", "safe.directory=" + str(path.resolve()), "-C", str(path)]
        actual = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            parser.error(f"{path}: expected {revision}, found {actual}")
        for extra in ([], ["--cached"]):
            subprocess.run([*git, "diff", "--quiet", *extra], check=True)
    output = args.output.resolve()
    receipt = output.with_suffix(output.suffix + ".json")
    if output.exists() or receipt.exists():
        parser.error("Output or receipt exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/kernels/jit/csrc/moe/xpu_compact"
    )
    import torch

    torch_root = Path(torch.__file__).resolve().parent
    prefill_source = source.parents[1] / "attention/xpu_dflash_prefill.cpp"
    head_source = source.parents[1] / "speculative/xpu_dflash_head"
    head_units = [
        head_source / name for name in ("rescore.cpp", "bounds.cpp", "conditional.cpp")
    ]
    includes = [
        torch_root / "include",
        torch_root / "include/torch/csrc/api/include",
        source,
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
        "-DDFLASH_NATIVE_WINDOW=" + str(args.window_size),
        "-fsycl-targets=spir64",
        "-Xspirv-translator",
        "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
        *["-I" + str(path.resolve()) for path in includes],
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        str(source / "compact.cpp"),
        str(source / "paged.cpp"),
        str(source / "routes.cpp"),
        str(source / "planner.cpp"),
        str(source / "metadata.cpp"),
        str(source / "dense_sequential.cpp"),
        str(source / "dense_small.cpp"),
        str(source / "dense_long.cpp"),
        str(prefill_source),
        *[str(path) for path in head_units],
        "-L" + str(torch_root / "lib"),
        "-Wl,-rpath," + str(torch_root / "lib"),
        "-ltorch_xpu",
        "-ltorch_cpu",
        "-lc10_xpu",
        "-lc10",
        "-o",
        str(output),
    ]
    print(shlex.join(command), flush=True)
    metadata = dict(
        command=command,
        window_size=args.window_size,
        torch_version=torch.__version__,
        torch_cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        kernel_revision=KERNEL_REV,
        sycl_tla_revision=TLA_REV,
        compiler=subprocess.check_output([args.compiler, "--version"], text=True),
        source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                source / "compact.cpp",
                source / "compact_kernel.hpp",
                source / "paged.cpp",
                source / "paged_kernel.hpp",
                source / "paged_mainloop.hpp",
                source / "routes.cpp",
                source / "planner.cpp",
                source / "metadata.cpp",
                source / "dflash_window.hpp",
                source / "dense_sequential.cpp",
                source / "dense_small.cpp",
                source / "dense_long.cpp",
                prefill_source,
                *head_units,
                head_source / "head_errors.hpp",
            )
        },
    )
    result = subprocess.run(command)
    metadata["returncode"] = result.returncode
    if result.returncode == 0:
        metadata["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(metadata, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
