"""Build precise SYCL GDN and integer input-copy kernels apart from expert DPAS."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="icpx")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    receipt = output.with_suffix(output.suffix + ".json")
    if output.exists() or receipt.exists():
        parser.error("Output or receipt exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[3] / (
        "python/sglang/kernels/jit/csrc/attention/xpu_dflash_gdn.cpp"
    )
    copy_source = source.with_name("xpu_dflash_small_copy.cpp")
    import torch

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
        str(source),
        str(copy_source),
        "-o",
        str(output),
    ]
    metadata = dict(
        command=command,
        torch_version=torch.__version__,
        compiler=subprocess.check_output([args.compiler, "--version"], text=True),
        source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source, copy_source)
        },
        scope="GDN ABI1/M4+M8 and small-copy ABI1; precise, independent of expert library",
    )
    result = subprocess.run(command)
    metadata["returncode"] = result.returncode
    if result.returncode == 0:
        metadata["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(metadata, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
