"""Build the opt-in B70 TP2 QK/RoPE or GDN norm/FP8 decode fusion."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=["qk-rope", "gdn-norm"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="icpx")
    args = parser.parse_args()
    name = {"qk-rope": "xpu_tp_qk_norm_rope.cpp",
            "gdn-norm": "xpu_tp_gdn_norm_quant.cpp"}[args.kernel]
    source = (Path(__file__).resolve().parents[3]
              / "python/sglang/kernels/jit/csrc/attention" / name)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.compiler, "-std=c++20", "-fsycl", "-fPIC", "-shared", "-O3",
               "-fp-model=precise", "-ffp-contract=off", "-fsycl-targets=spir64_gen",
               "-Xsycl-target-backend=spir64_gen", "-device bmg", str(source),
               "-o", str(output)]
    result = subprocess.run(command)
    receipt = dict(command=command, returncode=result.returncode,
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   contract="B70 TP2 B1 pure AR; BF16 rounding retained; opt-in")
    if result.returncode == 0:
        receipt["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
