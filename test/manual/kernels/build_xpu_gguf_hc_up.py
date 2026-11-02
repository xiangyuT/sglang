"""Build the optional B70 four-token GGUF HC down/low and up/mix kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler", default="icpx")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = root / "python/sglang/kernels/jit/csrc/quantization/xpu_gguf/hc_up_mix.cpp"
    down_source = source.with_name("hc_down_low.cpp")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.compiler, "-std=c++20", "-fsycl", "-fPIC", "-shared", "-O3",
        "-fp-model=precise", "-ffp-contract=off", "-fsycl-targets=spir64_gen",
        "-Xsycl-target-backend=spir64_gen",
        "-device bmg -options -cl-fp32-correctly-rounded-divide-sqrt",
        str(source), str(down_source), "-o", str(output),
    ]
    result = subprocess.run(command)
    receipt = dict(
        command=command, returncode=result.returncode,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sources_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source, down_source)
        },
    )
    if result.returncode == 0:
        receipt["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
