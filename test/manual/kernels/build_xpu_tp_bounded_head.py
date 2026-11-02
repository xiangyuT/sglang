"""Build the B70 TP2 Q4 filter and original BF16 exact-head library."""

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
        parser.error("Output or receipt exists; use a new path")
    source = Path(__file__).resolve().parents[3] / (
        "python/sglang/kernels/jit/csrc/gemm/xpu_tp_bounded_head.cpp"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.compiler,
        "-std=c++20",
        "-fsycl",
        "-fPIC",
        "-shared",
        "-O3",
        "-fp-model=precise",
        "-ffp-contract=off",
        "-fsycl-targets=spir64_gen",
        "-Xsycl-target-backend=spir64_gen",
        "-device bmg -options '-cl-fp32-correctly-rounded-divide-sqrt'",
        str(source),
        "-o",
        str(output),
    ]
    result = subprocess.run(command)
    metadata = dict(
        command=command,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        compiler=subprocess.check_output([args.compiler, "--version"], text=True),
        returncode=result.returncode,
        contract="B70 TP2 BF16 124160x2048; Q4 conservative filter only; original BF16 exact rows; split phases around native FP32 threshold MAX; SG32; raw weight retained",
    )
    if result.returncode == 0:
        metadata["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(metadata, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
