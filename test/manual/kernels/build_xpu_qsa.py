"""Build the optional BF16/INT8 QSA SYCL adapter inside the selected container."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="icpx")
    parser.add_argument(
        "--topk-only",
        action="store_true",
        help="Build only the optional stable TopK512 selector",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sources = root / "python/sglang/kernels/jit/csrc/attention/xpu_qsa"
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = [
        "qsa_bf16.cpp",
        "qsa_bf16_tp2.cpp",
        "qsa_bf16_tp4.cpp",
        "qsa_int8.cpp",
        "qsa_int8_tp2.cpp",
        "qsa_int8_tp4.cpp",
        "qsa_int8_store.cpp",
        "qsa_expand.cpp",
        "qsa_store.cpp",
        "qsa_rope.cpp",
        "qsa_metadata.cpp",
        "qsa_index_score.cpp",
        "qsa_topk.cpp",
    ]
    compiled_sources = [
        sources / name for name in (["qsa_topk.cpp"] if args.topk_only else names)
    ]
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
        *(str(source) for source in compiled_sources),
        "-o",
        str(output),
    ]
    result = subprocess.run(command)
    receipt = dict(
        topk_only=args.topk_only,
        command=command,
        returncode=result.returncode,
        sources={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in compiled_sources
        },
        contract="Strata chunk/merge attention and block-64 INT8 KV adapted to B70 physical slots; component numerical checks and model validation remain separate",
    )
    if result.returncode == 0:
        receipt["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
