# SPDX-License-Identifier: Apache-2.0
"""Norm hints must not cross token-row movement; residency is independent."""

from types import SimpleNamespace as NS

import pytest

from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.quantization.gguf_ep import ResidentEPGGUFMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def fixture_layer(ep, slots=None):
    method = object.__new__(ResidentEPGGUFMoEMethod)
    method.ep_size, method.real_experts = ep, 512 // ep
    method.num_experts = method.real_experts + 1
    method.cache_slots = method.num_experts if slots is None else slots
    method._gpu_cache = object()
    dispatcher = object.__new__(StandardDispatcher)
    dispatcher.num_local_shared_experts = 0
    norm = NS(_xpu_gguf_hc_enabled=True)
    layer = NS(
        mlp=NS(experts=NS(quant_method=method, dispatcher=dispatcher)),
        attn_hyper_connection=NS(hc_norm=norm),
        mlp_hyper_connection=NS(hc_norm=norm),
        _hc_norms=None,
    )
    parallel = NS(
        tp_size=ep, moe_ep_size=ep, moe_tp_size=1, attn_dp_size=1, attn_cp_size=1
    )
    return layer, parallel


@pytest.mark.parametrize("ep,slots", [(2, None), (2, 128), (4, None), (4, 64)])
@pytest.mark.parametrize(
    "blocker",
    [
        None,
        "opt_out",
        "dp",
        "cp",
        "scatter",
        "a2a",
        "dynamic_cache",
        "other_method",
        "other_dispatcher",
        "bad_budget",
        "shared_experts",
        "no_native_hc",
    ],
)
def test_norm_context_is_scoped_to_row_stable_ep(monkeypatch, ep, slots, blocker):
    from sglang.srt.models import qwen4_exp as model

    layer, parallel = fixture_layer(ep, slots)
    monkeypatch.setenv("SGLANG_XPU_GGUF_HC_TP_NORM_REUSE", "1")
    if blocker == "opt_out":
        monkeypatch.delenv("SGLANG_XPU_GGUF_HC_TP_NORM_REUSE")
    elif blocker == "dp":
        parallel.attn_dp_size = 2
    elif blocker == "cp":
        parallel.attn_cp_size = 2
    elif blocker == "dynamic_cache":
        layer.mlp.experts.quant_method._gpu_cache = None
    elif blocker == "other_method":
        layer.mlp.experts.quant_method = object()
    elif blocker == "other_dispatcher":
        layer.mlp.experts.dispatcher = object()
    elif blocker == "bad_budget":
        layer.mlp.experts.quant_method.cache_slots = 39
    elif blocker == "shared_experts":
        layer.mlp.experts.dispatcher.num_local_shared_experts = 1
    elif blocker == "no_native_hc":
        layer.attn_hyper_connection.hc_norm._xpu_gguf_hc_enabled = False
    monkeypatch.setattr(model, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        model, "get_attn_tp_context", lambda: NS(input_scattered=blocker == "scatter")
    )
    monkeypatch.setattr(
        model, "get_moe_a2a_backend", lambda: NS(is_none=lambda: blocker != "a2a")
    )
    normalized, next_norm = object(), object()
    with model.Qwen4ExpLayerExtensionMixin.hc_norms(
        layer, normalized, next_norm
    ) as norms:
        assert (layer._hc_norms is norms) == (blocker is None)
        assert norms.attn_input is normalized and norms.next_norm is next_norm
    assert layer._hc_norms is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
