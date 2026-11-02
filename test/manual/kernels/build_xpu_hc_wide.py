"""Build the optional B70 bounded-row HC mix/injection library."""

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
    folder = root / "python/sglang/kernels/jit/csrc/quantization/xpu_gguf"
    sources = [folder / name for name in ("hc_wide_down_low.cpp", "hc_wide_up_mix.cpp")]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.compiler, "-std=c++20", "-fsycl", "-fPIC", "-shared", "-O3",
        "-fp-model=precise", "-ffp-contract=off", "-fsycl-targets=spir64_gen",
        "-Xsycl-target-backend=spir64_gen",
        "-device bmg -options -cl-fp32-correctly-rounded-divide-sqrt",
        *(str(source) for source in sources), "-o", str(output),
    ]
    result = subprocess.run(command)
    receipt = dict(
        command=command, returncode=result.returncode,
        sources={source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
        contract="Original BF16 operands and four-row native per-token arithmetic. Python enables measured 8..24-row cases only. Model acceptance is separate.",
    )
    if result.returncode == 0:
        receipt["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
