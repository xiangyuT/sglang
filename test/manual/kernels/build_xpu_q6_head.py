"""Build optional exact-arithmetic B70 Q6_K vocabulary-head row sharing."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="icpx")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = (
        root / "python/sglang/kernels/jit/csrc/quantization/xpu_gguf/dense_q6_wide.cpp"
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
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
        "-device bmg -options -cl-fp32-correctly-rounded-divide-sqrt",
        str(source),
        "-o",
        str(output),
    ]
    result = subprocess.run(command)
    receipt = dict(
        command=command,
        returncode=result.returncode,
        sources={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [source, source.with_name("dequant.hpp")]
        },
        contract="Original Q6_K encoded SoA16 rows; token-independent original FP32 chains and reduction order; native1..16 rows; raw tails use existing4row DP4A adapter. Model acceptance remains separate.",
    )
    if result.returncode == 0:
        receipt["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
