"""Build the opt-in B70 SYCL quantization and sycl-tla small projection library.

Uses pinned local dependency checkouts, with no downloads or package changes.
Keep quantization in its own translation unit and preserve its compiler flags.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

KERNEL_REV = "028ac5a4e92a4addabeedac1a94374515f68e4a5"
TLA_REV = "7b56b936b87f799f80dde60c2fd8d11c1128cebb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--sycl-tla", type=Path, required=True)
    parser.add_argument("--compiler", default="icpx")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dependencies = []
    for path, revision in (
        (args.kernel_source, KERNEL_REV),
        (args.sycl_tla, TLA_REV),
    ):
        git = ["git", "-c", "safe.directory=" + str(path.resolve()), "-C", str(path)]
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        if head != revision:
            parser.error(f"{path}: expected {revision}, found {head}")
        for extra in ([], ["--cached"]):
            subprocess.run([*git, "diff", "--quiet", *extra], check=True)
        dependencies.append(dict(path=str(path.resolve()), revision=head))
    output = args.output.resolve()
    receipt = output.with_suffix(output.suffix + ".json")
    if output.exists() or receipt.exists():
        parser.error("Output or receipt exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[3] / (
        "python/sglang/kernels/jit/csrc/gemm/xpu_dflash_small_tla"
    )
    import torch

    torch_root = Path(torch.__file__).resolve().parent
    includes = [
        torch_root / "include",
        torch_root / "include/torch/csrc/api/include",
        args.kernel_source / "src/sycl",
        args.kernel_source / "src",
        args.kernel_source / "include",
        args.sycl_tla / "include",
        args.sycl_tla / "tools/util/include",
    ]
    units = [source / "quant.cpp", source / "projection.cpp"]
    command = [
        args.compiler,
        "-std=c++20",
        "-fsycl",
        "-fPIC",
        "-shared",
        "-O3",
        "-fp-model=precise",
        "-ffp-contract=off",
        "-fsycl-targets=spir64",
        "-Xspirv-translator",
        "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
        *["-I" + str(p.resolve()) for p in includes],
        *[str(p) for p in units],
        "-o",
        str(output),
    ]
    metadata = dict(
        command=command,
        dependencies=dependencies,
        compiler=subprocess.check_output([args.compiler, "--version"], text=True),
        torch_version=torch.__version__,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in units
        },
        scope="small-TLA ABI1/B70/raw-BF16 M4+M8; separate quant/prepare/DPAS kernels",
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
