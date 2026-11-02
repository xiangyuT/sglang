"""The real pool must allocate native records when it omits SSM snapshots."""
import os
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_GDN_SYCL_LIB"),
    reason="B70 GGUF GDN library required",
)


@pytest.mark.parametrize("radix", (False, True))
@pytest.mark.parametrize("tp", (1, 2, 4))
@pytest.mark.parametrize("batch", (1, 2, 8))
@torch.inference_mode()
def test_pool_owns_native_records_without_ssm_snapshots(monkeypatch, tp, batch, radix):
    from sglang.srt import runtime_context as runtime
    from sglang.srt.mem_cache.memory_pool import MambaPool

    if torch.xpu.get_device_properties().device_id != 0xE223:
        pytest.skip("B70")
    monkeypatch.setenv("SGLANG_XPU_GGUF_TP_EP", "1")
    monkeypatch.setenv("SGLANG_XPU_GGUF_GDN_SYCL", "1")
    monkeypatch.setenv("SGLANG_XPU_GGUF_GDN_SKIP_CHECKPOINTS", "1")
    monkeypatch.setattr(runtime, "get_spec", lambda: NS(
        speculative_algorithm="EAGLE", speculative_adaptive=False,
        speculative_eagle_topk=1,
    ))
    monkeypatch.setattr(runtime, "get_model", lambda: NS(quantization="gguf"))
    monkeypatch.setattr(runtime, "get_parallel", lambda: NS(tp_size=tp, moe_ep_size=tp, pp_size=1))
    monkeypatch.setattr(runtime, "get_schedule", lambda: NS(max_running_requests=batch))
    monkeypatch.setattr(runtime, "get_memory", lambda: NS(disable_radix_cache=not radix))
    monkeypatch.setattr(runtime, "get_exec", lambda: NS(mamba=NS(enable_mamba_extra_buffer=True)))
    heads = 48 // tp
    params = NS(
        is_kda=False,
        shape=NS(conv=[(10240 // tp, 3)], temporal=(heads, 128, 128),
                 disable_conv_window_dedup=False, conv_kernel=4),
        dtype=NS(conv=torch.bfloat16, temporal=torch.float32),
    )
    pool = MambaPool(size=batch, spec_state_size=batch, cache_params=params,
                     mamba_layer_ids=list(range(36)), device="xpu",
                     speculative_num_draft_tokens=4, speculative_eagle_topk=1)
    cache = pool.mamba_cache
    assert cache.intermediate_ssm is None
    assert cache.gguf_gdn_sycl is not None
    assert cache.gguf_gdn_sycl.matches_source(cache.temporal)
    assert cache.gguf_gdn_sycl.output.shape == (36, batch, 4, heads, 128)
    assert cache.gguf_gdn_sycl.records.shape == (36, batch, 82688 // tp)
    from sglang.kernels.ops.attention.gguf_gdn_sycl import scratch_bytes
    assert scratch_bytes(heads, batch) == sum(t.numel() * t.element_size() for t in cache.gguf_gdn_sycl.buffers)
    assert cache.intermediate_conv_window[0].shape[2] == 4
