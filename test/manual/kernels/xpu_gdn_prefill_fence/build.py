"""Build the existing GDN prefill fence shim without editing its source checkout."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgl-kernel-source", type=Path, required=True)
    parser.add_argument("--sycl-tla-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="icpx")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    provenance = json.loads((here / "provenance.json").read_text())
    source = args.sgl_kernel_source.resolve()
    tla = args.sycl_tla_source.resolve()
    output = args.output.resolve()
    include = output.with_name(output.name + ".includes")
    if output.exists() or include.exists():
        raise FileExistsError("Use a new output path and include directory")
    header = source / provenance["header"]
    original = header.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != provenance["original_header_sha256"]:
        raise ValueError("The GDN header does not match the pinned source revision")
    start = original.index("CUTE_DEVICE void chunk_fwd_o_kernel")
    end = original.index("template <typename T, typename StateTag>", start)
    body = original[start:end]
    old = "item.barrier(sycl::access::fence_space::local_space);"
    if body.count(old) != 3:
        raise ValueError("Unexpected GDN barrier layout")
    body = body.replace(old, "item.barrier(sycl::access::fence_space::global_and_local);")
    copy = "copy(copy_O2_c, tCrO2_c, tCgO2_c);"
    if body.count(copy) != 1:
        raise ValueError("Unexpected GDN output-copy layout")
    body = body.replace(copy, copy + "\n      item.barrier(sycl::access::fence_space::global_and_local);")
    patched = original[:start] + body + original[end:]
    if hashlib.sha256(patched.encode()).hexdigest() != provenance["patched_header_sha256"]:
        raise ValueError("The prepared header does not match the recorded fence implementation")
    target = include / "sycl/kernels/gdn_attn/chunk_gated_delta_rule_kernels.hpp"
    target.parent.mkdir(parents=True)
    target.write_text(patched)
    if args.prepare_only:
        print(json.dumps({"prepared_header": str(target), "sha256": provenance["patched_header_sha256"]}))
        return
    import torch

    torch_root = Path(torch.__file__).resolve().parent
    command = [args.compiler, "-std=c++20", "-fsycl", "-fPIC", "-shared", "-O3",
               "-fsycl-targets=spir64", "-Xspirv-translator",
               "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
               "-DCUTLASS_ENABLE_SYCL", "-DSYCL_INTEL_TARGET=20",
               "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    directories = [torch_root / "include", torch_root / "include/torch/csrc/api/include",
                   include, source / "src", source / "src/sycl", source / "include",
                   tla / "include", tla / "tools/util/include"]
    command += ["-I" + str(path) for path in directories]
    command += [str(here / "chunk_bridge.cpp"), "-L" + str(torch_root / "lib"),
                "-Wl,-rpath," + str(torch_root / "lib"),
                "-ltorch_xpu", "-ltorch_cpu", "-lc10_xpu", "-lc10", "-o", str(output)]
    result = subprocess.run(command)
    receipt = {"command": command, "returncode": result.returncode, "provenance": provenance}
    if result.returncode == 0:
        receipt["library_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result.check_returncode()

if __name__ == "__main__":
    main()
