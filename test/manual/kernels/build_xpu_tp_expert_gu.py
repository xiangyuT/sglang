"""Build optional B70 TP2/TP4 and EP2 packed-word gate/up projection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compiler',default='icpx')
    args=p.parse_args()
    source=Path(__file__).resolve().parents[3]/'python/sglang/kernels/jit/csrc/moe/xpu_tp_expert_gu.cpp'
    header=source.parent.parent/'gemm/xpu_tp_fp8_codec.hpp'
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    cmd=[args.compiler,'-std=c++20','-fsycl','-fPIC','-shared','-O3',
         '-fp-model=precise','-ffp-contract=off','-fsycl-targets=spir64_gen',
         '-Xsycl-target-backend=spir64_gen','-device bmg -options -cl-fp32-correctly-rounded-divide-sqrt',str(source),'-o',str(output)]
    process=subprocess.run(cmd)
    receipt=dict(command=cmd,returncode=process.returncode,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
        contract='Original FP8/block128 scales and truncated-BF16 dequant; FP32 tree; BF16 gate/up; I512 optionally fuses BF16 SiLU intermediate and FP32 multiply after BF16 gate/up; other paths retain native SiLU')
    if process.returncode==0:receipt['library_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    process.check_returncode()


if __name__=='__main__':main()
