"""Build B70 resident or paged expert workspace preparation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', default='icpx')
    parser.add_argument('--paged', action='store_true',
                        help='Build logical-dummy clipping over the existing GPU pager')
    args = parser.parse_args()
    filename = 'paged_prepare.cpp' if args.paged else 'resident_prepare.cpp'
    source = (Path(__file__).resolve().parents[3]
              / 'python/sglang/kernels/jit/csrc/quantization/xpu_gguf' / filename)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.compiler, '-std=c++20', '-fsycl', '-fPIC', '-shared', '-O3',
               '-fp-model=precise', '-ffp-contract=off', '-fsycl-targets=spir64_gen',
               '-Xsycl-target-backend=spir64_gen',
               '-device bmg -options -cl-fp32-correctly-rounded-divide-sqrt', str(source),
               '-lze_loader', '-o', str(output)]
    p = subprocess.run(command)
    receipt = dict(command=command, returncode=p.returncode,
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   contract='B70 fixed identity resident EP2/EP4; rows1..4/topk1..10; IDs32/64; final zero slot; same-stream FP32 scratch H2560/I640; optional finiteQ8_1 input producer with production divide rounding; hardware validation separately')
    if args.paged:
        receipt['contract'] = ('B70 existing GPU-cache grouping; logical dummy may move '
                               'between physical slots; rows1..32/topk1..10 H2560/I640; '
                               'same-stream scratch clear, optional per-shape FP32 arena '
                               'with one clear per byte, and finiteQ8_1 input; '
                               'no placement/LRU/encoded-weight mutation; hardware validation separately')
    if p.returncode == 0:
        receipt['library_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.build.json').write_text(json.dumps(receipt, indent=2)+'\n')
    p.check_returncode()


if __name__ == '__main__':
    main()
