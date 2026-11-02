"""Build the opt-in B70 small-message peer collective (TP2/TP4)."""
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
    source = (Path(__file__).resolve().parents[3]
              / 'python/sglang/kernels/jit/csrc/distributed/xpu_peer_allreduce.cpp')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.compiler, '-std=c++20', '-fsycl', '-fPIC', '-shared', '-O3',
               '-fp-model=precise', '-ffp-contract=off', '-fsycl-targets=spir64_gen',
               '-Xsycl-target-backend=spir64_gen', '-device bmg', str(source),
               '-lze_loader', '-o', str(output)]
    p = subprocess.run(command)
    receipt = dict(command=command, returncode=p.returncode,
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   contract='BF16 aligned8..2097152elements divisible8; opaque allgather4..2097152bytes divisible4; TP2/TP4 local B70; rank-ordered FP32 sum; hardware validation recorded separately')
    if p.returncode == 0:
        receipt['library_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.build.json').write_text(json.dumps(receipt, indent=2)+'\n')
    p.check_returncode()


if __name__ == '__main__':
    main()
