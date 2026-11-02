"""Build the optional B70 TP/EP expert-down library inside a SYCL environment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', default='icpx')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3] / 'python/sglang/kernels/jit/csrc/moe/xpu_tp_expert_down.cpp'
    header = source.parent.parent / 'gemm/xpu_tp_fp8_codec.hpp'
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.compiler, '-std=c++20', '-fsycl', '-fPIC', '-shared', '-O3',
               '-fp-model=precise', '-ffp-contract=off', '-fsycl-targets=spir64_gen',
               '-Xsycl-target-backend=spir64_gen',
               '-device bmg -options -cl-fp32-correctly-rounded-divide-sqrt',
               str(source), '-o', str(output)]
    process = subprocess.run(command)
    receipt = dict(command=command, returncode=process.returncode,
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
                   numerical_contract='Native GU/SiLU retained; BF16 weight truncation; FP32 down tree; BF16 route outputs; FP32 weighted reduction')
    if process.returncode == 0:
        receipt['library_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.build.json').write_text(json.dumps(receipt, indent=2)+'\n')
    process.check_returncode()


if __name__ == '__main__':
    main()
