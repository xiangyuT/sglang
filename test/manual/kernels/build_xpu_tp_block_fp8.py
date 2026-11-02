"""Build the optional B70 TP2/TP4 projection library inside a SYCL environment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', default='icpx')
    parser.add_argument('--auxiliary', action='store_true',
                        help='Build TP shared/output projections instead of input projections')
    args = parser.parse_args()
    name = 'xpu_tp_aux_block_fp8.cpp' if args.auxiliary else 'xpu_tp_block_fp8.cpp'
    source = Path(__file__).resolve().parents[3] / 'python/sglang/kernels/jit/csrc/gemm' / name
    header = source.with_name('xpu_tp_fp8_codec.hpp')
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
                   numerical_contract=('FP8 input/scales retained; shape-specific FP16/FP32 dequant; FP32 tree; BF16 output; not bit-equivalent to native scaled_mm' if args.auxiliary else 'FP8 input/scales retained; FP16 dequant; FP32 tree; BF16 output; not bit-equivalent to native scaled_mm'))
    if process.returncode == 0:
        receipt['library_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.build.json').write_text(json.dumps(receipt, indent=2)+'\n')
    process.check_returncode()


if __name__ == '__main__':
    main()
