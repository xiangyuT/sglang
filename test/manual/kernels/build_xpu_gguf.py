"""Build the optional B70 GGUF decoder port inside the selected SYCL container."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="icpx")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sources = root / "python/sglang/kernels/jit/csrc/quantization/xpu_gguf"
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
        str(sources / "dequant.cpp"),
        str(sources / "matvec.cpp"),
        str(sources / "routes.cpp"),
        str(sources / "routes_general.cpp"),
        str(sources / "scatter.cpp"),
        str(sources / "expert_q8.cpp"),
        str(sources / "expert_q8_quantize.cpp"),
        str(sources / "moe_q8.cpp"),
        str(sources / "mixed_packets.cpp"),
        str(sources / "iq4_expert_soa.cpp"),
        str(sources / "iq3_expert_soa.cpp"),
        str(sources / "iq2s_pvr.cpp"),
        str(sources / "iq2s_plane_split.cpp"),
        str(sources / "lowbit_gu.cpp"),
        str(sources / "q2_pvr.cpp"),
        str(sources / "q2_raw.cpp"),
        str(sources / "expert_copy.cpp"),
        str(sources / "dense_q8.cpp"),
        str(sources / "dense_q8_soa.cpp"),
        str(sources / "dense_q6_soa.cpp"),
        str(sources / "dense_q6_wide.cpp"),
        str(sources / "iq3_xmx.cpp"),
        str(sources / "dense_dp4a.cpp"),
        str(sources / "iq4_s4.cpp"),
        str(sources / "hc_esimd.cpp"),
        str(sources / "moe_quant_fused.cpp"),
        str(sources / "top10.cpp"),
        str(sources / "bf16_small.cpp"),
        str(sources / "dense_dp4a_bf16.cpp"),
        str(sources / "iq4_s4_bf16.cpp"),
        str(sources / "iq4_s4_wg_bf16.cpp"),
        str(sources / "iq3_packed_bf16.cpp"),
        str(sources / "dense_q8_bf16.cpp"),
        str(sources / "hc_parallel.cpp"),
        str(sources / "top10_norm.cpp"),
        str(sources / "mtp_top1.cpp"),
        str(sources / "gpu_cache.cpp"),
        str(sources / "hc.cpp"),
        "-o",
        str(output),
    ]
    result = subprocess.run(command)
    receipt = {
        "command": command,
        "returncode": result.returncode,
        "sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources.iterdir())
            if p.is_file()
        },
        "contract": "Ported Strata GGUF decoders and optional Q8_1/lane8 expert dots; original weight bytes; FP32 expert outputs; component checks do not establish model correctness or performance",
    }
    if result.returncode == 0:
        receipt["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
