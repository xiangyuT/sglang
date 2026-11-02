"""Build the independent B70 FP32 greedy packet kernel; preserve precise math."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[3]
source = root / 'python/sglang/kernels/jit/csrc/distributed/xpu_gguf_tp_greedy.cpp'
args.output.parent.mkdir(parents=True, exist_ok=True)
cmd = ['/opt/intel/oneapi/compiler/latest/bin/icpx', '-std=c++20', '-fsycl',
       '-shared', '-fPIC', '-O3', '-fp-model=precise', '-ffp-contract=off',
       '-fsycl-targets=spir64_gen', '-Xsycl-target-backend=spir64_gen',
       '-device bmg -options -cl-fp32-correctly-rounded-divide-sqrt', str(source),
       '-o', str(args.output)]
result = subprocess.run(cmd)
receipt = dict(command=cmd, returncode=result.returncode,
               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               compiler=subprocess.check_output([cmd[0], '--version'], text=True),
               scope='FP32 local argmax + byte packet + exact greedy carrier for1..32rows; existing TP2/4 transport')
if result.returncode == 0:
    receipt['library_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
args.output.with_suffix('.build.json').write_text(json.dumps(receipt, indent=2) + '\n')
result.check_returncode()
