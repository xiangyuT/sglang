"""Build Strata's optional IQ4_XS/Q8_1 MMVQ port for B70."""

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
    directory = root / "python/sglang/kernels/jit/csrc/quantization/xpu_iq4"
    source = directory / "iq4_mmvq.cpp"
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
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        strata_source="c3ed1d047768d94f43d7384ccf740a214f5a53dd",
        numerical_contract="IQ4_XS original bytes; finite Q8_1 activation quantization; FP32 output",
    )
    if result.returncode == 0:
        receipt["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
