"""Launch with the recorded process-local GDN prefill fence compatibility shim."""
import ctypes
import os
import runpy

def install():
    import torch  # noqa: F401 - Load the Torch libraries before the C++ shim.

    path = os.environ["SGLANG_XPU_GDN_PREFILL_FENCE_LIB"]
    library = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    library.gguf_gdn_fence_calls.restype = ctypes.c_uint64
    # The shim must be global before the installed native GDN module is loaded.
    import sgl_kernel
    import sgl_kernel.gdn_attn as module
    original = module.gdn_attention
    logged = False

    def checked_gdn(*args, **kwargs):
        nonlocal logged
        value = original(*args, **kwargs)
        if kwargs.get("num_prefills", 0) > 0:
            if library.gguf_gdn_fence_calls() == 0:
                raise RuntimeError("The native GDN fence replacement was not invoked")
            if not logged:
                print("[gguf-gdn-fence] BF16/FP32-state prefill replacement invoked", flush=True)
                logged = True
        return value

    module.gdn_attention = checked_gdn
    sgl_kernel.gdn_attention = checked_gdn

install()

if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name="__main__")
