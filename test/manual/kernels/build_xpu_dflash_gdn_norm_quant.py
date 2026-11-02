"""Build the precise B70 M8 GDN norm/block128 quant library."""

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
    source = Path(__file__).resolve().parents[3] / (
        "python/sglang/kernels/jit/csrc/attention/xpu_dflash_gdn_norm_quant.cpp"
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
        "-fsycl-targets=spir64",
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
        scope="GDN norm/quant ABI1, M8 head128 SG2 GRF256, BF16 boundary; precise",
    )
    if result.returncode == 0:
        metadata["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(metadata, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
