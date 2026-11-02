# SPDX-License-Identifier: Apache-2.0
"""Build the opt-in B70 grouped FP16 expert-prefill port."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', default='icpx')
    parser.add_argument('--f16-layer', action='store_true', help='Build native IQ3_S/IQ4_XS + Q8_0 and Q2_0 expert layers')
    parser.add_argument('--int8', action='store_true', help='Build the fused IQ3_S/IQ4_NL INT8 chain')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[3] / 'python/sglang/kernels/jit/csrc/quantization/xpu_gguf'
    if args.int8 and args.f16_layer:
        parser.error('--int8 and --f16-layer select different libraries')
    source = directory / ('prefill_i8.cpp' if args.int8 else 'prefill_layer.cpp' if args.f16_layer else 'prefill_moe.cpp')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.compiler, '-std=c++20', '-fsycl', '-fPIC', '-shared', '-O3',
               '-fp-model=precise', '-ffp-contract=off', '-fsycl-targets=spir64_gen',
               '-Xsycl-target-backend=spir64_gen',
               '-device bmg -options -cl-fp32-correctly-rounded-divide-sqrt',
               str(source), '-o', str(output)]
    if args.int8 or args.f16_layer:
        command.append("-ldl")
    if args.int8 or args.f16_layer:
        command.append("-pthread")
    result = subprocess.run(command)
    receipt = dict(command=command, returncode=result.returncode,
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if args.f16_layer:
        receipt['headers'] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                              for name in ('prefill_moe.cpp', 'prefill_copy.hpp', 'prefill_pack.hpp', 'dequant.hpp', 'iq3_expert_soa.hpp', 'ggml-common.h')}
    if args.int8:
        receipt['headers'] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                              for name in ('prefill_i8.hpp', 'iq_tables.inc', 'prefill_copy.hpp', 'prefill_pack.hpp')}
    if result.returncode == 0:
        receipt['library_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    result.check_returncode()


if __name__ == '__main__':
    main()
