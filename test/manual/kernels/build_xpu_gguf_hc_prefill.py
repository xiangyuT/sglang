"""Build isolated B70 prefill HC normalization with Torch reduction ordering."""
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
    directory = Path(__file__).resolve().parents[3] / 'python/sglang/kernels/jit/csrc/quantization/xpu_gguf'
    source = directory / 'hc_prefill.cpp'
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.compiler, '-std=c++20', '-fsycl', '-fPIC', '-shared', '-O3',
               '-fp-model=precise', '-ffp-contract=off', '-fsycl-targets=spir64_gen',
               '-Xsycl-target-backend=spir64_gen',
               '-device bmg -options -cl-fp32-correctly-rounded-divide-sqrt',
               str(source), '-o', str(output)]
    result = subprocess.run(command)
    receipt = dict(command=command, returncode=result.returncode,
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   header_sha256=hashlib.sha256((directory / 'hc_esimd.cpp').read_bytes()).hexdigest())
    if result.returncode == 0:
        receipt['library_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    result.check_returncode()


if __name__ == '__main__':
    main()
