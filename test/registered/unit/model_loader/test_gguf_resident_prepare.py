# SPDX-License-Identifier: Apache-2.0
"""Stream ownership and fail-closed guards; hardware math tested separately."""

from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from sglang.kernels.ops.quantization import gguf_resident_prepare as resident
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def route(library):
    count = 40
    data = torch.empty(6 * count + 3, dtype=torch.int32)
    return resident.ResidentRoutes(
        data,
        data[:count],
        data[count : 2 * count],
        data[2 * count : 3 * count + 1],
        data[3 * count + 1 : 4 * count + 1],
        data[4 * count + 1 : 5 * count + 1],
        count,
        torch.zeros((4, 10), dtype=torch.int32),
        128,
        library,
    )


def scratch():
    return {
        name: torch.empty((40, width), dtype=torch.float32)
        for name, width in (("gate", 640), ("up", 640), ("down", 2560))
    }


def test_actual_consumer_stream_workspace_is_passed_to_native(monkeypatch):
    calls = []
    lib = SimpleNamespace(resident_prepare=lambda *args: calls.append(args) or 0)
    value = route(lib)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: False)
    first, second = scratch(), scratch()
    for queue, workspace in ((11, first), (22, second)):
        value.prepare_resident(workspace, SimpleNamespace(sycl_queue=queue), 128)
        args = calls[-1]
        assert args[0] == queue
        assert args[3:6] == tuple(
            workspace[n].data_ptr() for n in ("gate", "up", "down")
        )
    assert calls[0][3:6] != calls[1][3:6]


def test_changed_zero_slot_is_rejected_before_native():
    with pytest.raises(ValueError, match="zero slot changed"):
        route(object()).prepare_resident(scratch(), object(), 256)


@pytest.mark.parametrize("field", ["gate", "up", "down"])
def test_wrong_scratch_is_rejected_before_native(field):
    values = scratch()
    values[field] = values[field][:39]
    with pytest.raises(ValueError, match="actual-stream FP32 scratch"):
        route(object()).prepare_resident(values, object(), 128)


def test_unset_feature_requires_no_cache_or_device(monkeypatch):
    monkeypatch.delenv("SGLANG_XPU_GGUF_RESIDENT_PREP_LIB", raising=False)
    assert resident.configure(None, None) is None


def test_configure_during_capture_is_rejected(monkeypatch):
    monkeypatch.setenv("SGLANG_XPU_GGUF_RESIDENT_PREP_LIB", "unused.so")
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before capture"):
        resident.configure(None, None)


def test_mapped_routes_keep_the_consumers_map_and_stream_scratch(monkeypatch):
    calls = []
    lib = SimpleNamespace(
        resident_prepare=lambda *args: pytest.fail("local preparation selected"),
        resident_prepare_mapped=lambda *args: calls.append(args) or 0,
    )
    mapping = torch.arange(512, dtype=torch.int32)
    value = replace(route(lib), local_mapping=mapping)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: False)
    for queue in (11, 22):
        workspace = scratch()
        value.prepare_resident(workspace, SimpleNamespace(sycl_queue=queue), 128)
        args = calls[-1]
        assert args[:4] == (
            queue, value.ids.data_ptr(), mapping.data_ptr(), value.data.data_ptr()
        )
        assert args[4:7] == tuple(
            workspace[n].data_ptr() for n in ("gate", "up", "down")
        )
        assert args[7:] == (4, 10, 128, 0, 512)


@pytest.mark.parametrize("mapping", [torch.zeros(511, dtype=torch.int32),
                                     torch.zeros(512, dtype=torch.int64),
                                     torch.zeros(1024, dtype=torch.int32)[::2]])
def test_mapped_routes_reject_an_incompatible_map(mapping):
    lib = SimpleNamespace(resident_prepare=lambda *args: None,
                          resident_prepare_mapped=lambda *args: pytest.fail("native called"))
    with pytest.raises(ValueError, match="E512 expert map"):
        replace(route(lib), local_mapping=mapping).prepare_resident(
            scratch(), SimpleNamespace(sycl_queue=11), 128
        )


def test_dispatch_rejects_another_preparations_mapping():
    preparation = resident.ResidentPreparation(None, 128, object())
    output = SimpleNamespace(xpu_gguf_resident_routing=(object(), object()))
    with pytest.raises(ValueError, match="does not belong"):
        preparation.routes_dispatch(output)


def test_legacy_dispatch_keeps_local_ids():
    preparation = resident.ResidentPreparation(None, 128, object())
    ids = torch.zeros((1, 10), dtype=torch.int64)
    calls = []
    preparation.routes = lambda value, **kw: calls.append((value, kw)) or "routes"
    assert preparation.routes_dispatch(SimpleNamespace(
        topk_output=SimpleNamespace(topk_ids=ids))) == "routes"
    assert calls[0][0] is ids and calls[0][1] == {"local_mapping": None}


@pytest.mark.parametrize("mapped", [False, True])
def test_input_producer_uses_the_selected_consumer_workspace(monkeypatch, mapped):
    calls = []
    lib = SimpleNamespace(
        resident_prepare=lambda *args: pytest.fail("separate preparation selected"),
        resident_prepare_mapped=lambda *args: pytest.fail("separate map selected"),
        resident_prepare_q8=lambda *args: calls.append(args) or 0,
    )
    value = route(lib)
    if mapped:
        value = replace(value, local_mapping=torch.arange(512, dtype=torch.int32))
    x = torch.zeros((4, 2560), dtype=torch.bfloat16)
    live_workspaces = []
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: False)
    for queue in (11, 22):
        workspace = scratch()
        workspace.update(
            x=torch.empty((4, 2560)),
            scratch=torch.empty(4 * 80 * 36, dtype=torch.uint8),
        )
        live_workspaces.append(workspace)
        assert value.prepare_resident(
            workspace, SimpleNamespace(sycl_queue=queue), 128, x=x
        )
        args = calls[-1]
        assert args[:4] == (
            queue,
            value.ids.data_ptr(),
            value.local_mapping.data_ptr() if mapped else None,
            value.data.data_ptr(),
        )
        assert args[4:10] == tuple(
            workspace[n].data_ptr() for n in ("gate", "up", "down")
        ) + (x.data_ptr(), workspace["x"].data_ptr(), workspace["scratch"].data_ptr())
        assert args[10:] == (4, 10, 128, 0, 512)
    assert calls[0][8:10] != calls[1][8:10]


@pytest.mark.parametrize(
    "bad",
    [
        "input_dtype",
        "input_shape",
        "value_dtype",
        "value_shape",
        "quant_size",
        "quant_dtype",
    ],
)
def test_input_producer_rejects_incompatible_workspace_before_native(bad):
    lib = SimpleNamespace(
        resident_prepare=lambda *args: None,
        resident_prepare_q8=lambda *args: pytest.fail("native called"),
    )
    x = torch.zeros((4, 2560), dtype=torch.bfloat16)
    workspace = scratch()
    workspace.update(
        x=torch.empty((4, 2560)), scratch=torch.empty(4 * 80 * 36, dtype=torch.uint8)
    )
    if bad == "input_dtype":
        x = x.float()
    elif bad == "input_shape":
        x = x[:3]
    elif bad == "value_dtype":
        workspace["x"] = workspace["x"].bfloat16()
    elif bad == "value_shape":
        workspace["x"] = workspace["x"][:3]
    elif bad == "quant_size":
        workspace["scratch"] = workspace["scratch"][:-1]
    elif bad == "quant_dtype":
        workspace["scratch"] = workspace["scratch"].to(torch.int32)
    with pytest.raises(ValueError, match="actual-stream BF16/FP32/Q8"):
        route(lib).prepare_resident(workspace, SimpleNamespace(sycl_queue=11), 128, x=x)


def test_old_library_cannot_claim_to_have_produced_input():
    lib = SimpleNamespace(resident_prepare=lambda *args: pytest.fail("native called"))
    with pytest.raises(ValueError, match="Rebuild resident preparation"):
        route(lib).prepare_resident(
            scratch(),
            SimpleNamespace(sycl_queue=11),
            128,
            x=torch.zeros((4, 2560), dtype=torch.bfloat16),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
