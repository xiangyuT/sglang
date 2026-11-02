"""Queued native H2D lifetime, source slices and expert cache interoperability."""

from collections import OrderedDict
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.kernels.ops.quantization.gguf_expert_copy import (
    _CopyRing,
    make_expert_copies,
)
from sglang.kernels.ops.quantization.gguf_iq2s_pvr import (
    group_iq2s_routes,
    repack_iq2s_cache,
)
from sglang.kernels.ops.quantization.gguf_sycl import _library
from sglang.srt.layers.quantization.qwen4_gguf_xpu import CompressedGGUFMoEMethod
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES
from sglang.srt.model_loader.gguf_rows import (
    CompressedGGUFExpertSlice,
    CompressedGGUFRows,
)

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB"),
    reason="requires the separately built B70 GGUF library",
)


def sources(kind, sliced=False):
    result = []
    for seed, (dtype, rows, width) in enumerate(
        [(kind, 640, 2560), (kind, 640, 2560), (42, 2560, 640)]
    ):
        block, size = BLOCK_GEOMETRY[GGML_TYPES[dtype]]
        raw = np.random.default_rng(seed).integers(
            0, 256, (6 * rows * width // block, size), dtype=np.uint8
        )
        if dtype == 29:
            scales = raw[:, 48:].copy().view(np.uint16)
            for i in range(4):
                scales[:, i] = (scales[:, i] & 0xFFF) | (
                    ((0x3400 >> (4 * i)) & 15) << 12
                )
            raw[:, 48:] = scales.view(np.uint8)
        else:
            raw[:, :2] = np.array([0.25], dtype=np.float16).view(np.uint8)
        value = CompressedGGUFRows.__new__(CompressedGGUFRows)
        value.width, value.rows, value.kind = width, rows * 6, dtype
        value.row_bytes = width // block * size
        value._mapping = raw.reshape(value.rows, value.row_bytes)
        result.append(value)
    if sliced:
        # Original MTP storage has [expert, gate rows then up rows].
        parent = CompressedGGUFRows.__new__(CompressedGGUFRows)
        parent.width, parent.rows, parent.kind = 2560, 1280 * 6, kind
        parent.row_bytes = result[0].row_bytes
        parent.tensor = SimpleNamespace(shape=(2560, 1280, 6))
        parent._mapping = np.stack(
            [
                np.concatenate(
                    (
                        result[0]._mapping[i * 640 : (i + 1) * 640],
                        result[1]._mapping[i * 640 : (i + 1) * 640],
                    )
                )
                for i in range(6)
            ]
        ).reshape(parent.rows, parent.row_bytes)
        result[:2] = [
            CompressedGGUFExpertSlice(parent, 0, 640),
            CompressedGGUFExpertSlice(parent, 640, 1280),
        ]
    for source in result:
        mapping = (
            source.source._mapping
            if isinstance(source, CompressedGGUFExpertSlice)
            else source._mapping
        )
        mapping.flags.writeable = False
    return result


def method_layer(monkeypatch, values, native, slots=3):
    monkeypatch.setenv("SGLANG_XPU_GGUF_NATIVE_STAGE", "1" if native else "0")
    layer = torch.nn.Module()
    method = CompressedGGUFMoEMethod(slots, decode_q8=True)
    method.create_weights(layer, 6, 2560, 640, torch.bfloat16)
    method.bind_rows(layer, values, torch.device("xpu:0"))
    method.fused_q8 = True
    method.iq2s_pvr = values[0].kind == 22
    assert (method._native_staging is not None) == native
    return method, layer


@pytest.mark.parametrize(
    "kind,sliced", [(16, False), (22, False), (29, False), (42, False), (42, True)]
)
def test_raw_bytes_lru_and_queued_packed_transitions(monkeypatch, kind, sliced):
    values = sources(kind, sliced)
    method, layer = method_layer(monkeypatch, values, True)
    originals = [
        source.source._mapping
        if isinstance(source, CompressedGGUFExpertSlice)
        else source._mapping
        for source in values
    ]
    before = [array.copy() for array in originals]
    snapshots = []
    for repeat in range(4):
        for index, selected in enumerate(
            ([1, 2, 3], [0, 1], [1, 2, 4], [0, 4], [4, 5])
        ):
            method._ensure_group(
                layer, selected, packed_gu=(kind == 22 and (repeat + index) % 2 == 0)
            )
            copies = [
                getattr(layer, "gguf_" + name).clone()
                for name in ("gate", "up", "down")
            ]
            if kind == 22:
                repack_iq2s_cache(
                    copies[0],
                    copies[1],
                    method._iq2s_packed_slots,
                    to_packed=False,
                    down=copies[2],
                )
            snapshots.append((copies, OrderedDict(method.cache)))
    for copies, cache in snapshots:
        for copy, source in zip(copies, values):
            actual = copy.cpu()
            rows = source.rows // 6
            for expert, slot in cache.items():
                expected = source.packed_row_range(expert * rows, (expert + 1) * rows)
                assert torch.equal(actual[slot], expected.reshape(-1))
    assert method._native_staging.submissions == method.misses
    assert (
        method.transferred_bytes == method.misses * method._native_staging.expert_bytes
    )
    assert all(np.array_equal(a, b) for a, b in zip(originals, before))


@pytest.mark.parametrize("kind", [16, 22, 29, 42])
def test_full_expert_outputs_after_repeated_replacements(monkeypatch, kind):
    values = sources(kind)
    original, old_layer = method_layer(monkeypatch, values, False)
    method, layer = method_layer(monkeypatch, values, True)
    g = torch.Generator().manual_seed(42)
    for ids in (
        [[2, 0]],
        [[2, 3], [1, 0], [3, 2], [0, 1]],
        [[i % 6, (i + 1) % 6] for i in range(8)],
        [[4, 5], [5, 2], [2, 4], [5, 4]],
    ):
        with torch.inference_mode():
            x = (torch.randn(len(ids), 2560, generator=g) * 0.01).to(
                device="xpu", dtype=torch.bfloat16
            )
            order = torch.tensor(ids, device="xpu", dtype=torch.int32)
            weights = torch.tensor([[0.3, 0.7]], device="xpu").repeat(len(ids), 1)
            expected = original._apply_routes.run_eager(old_layer, x, weights, order)
            actual = method._apply_routes.run_eager(layer, x, weights, order)
            assert torch.equal(
                actual.cpu().view(torch.int16), expected.cpu().view(torch.int16)
            )
    assert method._native_staging.submissions == method.misses


@pytest.mark.parametrize("kind", [16, 22, 29, 42])
def test_cache_growth_preserves_slots_packing_and_native_destinations(
    monkeypatch, kind
):
    values = sources(kind)
    method, layer = method_layer(monkeypatch, values, True)
    reference, reference_layer = method_layer(monkeypatch, values, False, slots=6)
    method._ensure_group(layer, [1, 2, 3], packed_gu=kind == 22)
    names = ("gate", "up", "down")
    before = [getattr(layer, "gguf_" + name).clone() for name in names]
    pointers = [getattr(layer, "gguf_" + name).data_ptr() for name in names]
    cache, markers = OrderedDict(method.cache), set(method._iq2s_packed_slots)
    counters = (method.hits, method.misses, method.transferred_bytes)
    added = method.grow_cache(layer, 6)
    assert added == sum(value.numel() for value in before)
    assert method.cache_slots == 6
    assert method.cache == cache and method._iq2s_packed_slots == markers
    assert (method.hits, method.misses, method.transferred_bytes) == counters
    for name, value, pointer in zip(names, before, pointers):
        actual = getattr(layer, "gguf_" + name)
        assert actual.data_ptr() != pointer
        assert torch.equal(actual[:3], value)
    assert method.grow_cache(layer, 6) == 0
    with pytest.raises(ValueError):
        method.grow_cache(layer, 3)
    with pytest.raises(ValueError):
        method.grow_cache(layer, 7)
    generator = torch.Generator().manual_seed(42)
    for tokens in (4, 8, 1, 4):
        x = (torch.randn(tokens, 2560, generator=generator) * 0.01).to(
            device="xpu", dtype=torch.bfloat16
        )
        ids = torch.tensor(
            [[i % 6, (i + 4) % 6] for i in range(tokens)],
            device="xpu",
            dtype=torch.int32,
        )
        weights = torch.tensor([[0.3, 0.7]], device="xpu").repeat(tokens, 1)
        expected = reference._apply_routes.run_eager(reference_layer, x, weights, ids)
        actual = method._apply_routes.run_eager(layer, x, weights, ids)
        assert torch.equal(
            actual.cpu().view(torch.int16), expected.cpu().view(torch.int16)
        )
    assert method._native_staging.submissions == method.misses


def test_small_ring_wrap_and_close_before_cpu_buffer_release():
    library = _library(os.environ["SGLANG_XPU_GGUF_SYCL_LIB"])
    all_snapshots = []
    for _ in range(2):
        stream = torch.xpu.Stream()
        with torch.xpu.stream(stream):
            ring = _CopyRing(library, stream, slots=2)
            buffers = [
                torch.empty(32, device="xpu", dtype=torch.uint8) for _ in range(3)
            ]
            raw = np.empty((3, 32), dtype=np.uint8)
            try:
                for step in range(144):
                    raw[:] = np.arange(32, dtype=np.uint8)[None] + step
                    status = library.xpu_gguf_copy_expert(
                        ring.handle,
                        *[x.data_ptr() for x in buffers],
                        *[raw[i].ctypes.data for i in range(3)],
                        32,
                        32,
                        32,
                    )
                    assert status == 0
                    all_snapshots.append(([x.clone() for x in buffers], raw.copy()))
            finally:
                ring.close()
            assert ring.handle is None
            ring.storage = None
    for copies, expected in all_snapshots:
        for copy, reference in zip(copies, expected):
            assert np.array_equal(copy.cpu().numpy(), reference)


def test_unsupported_sources_and_invalid_indices(monkeypatch):
    values = sources(22)
    method, layer = method_layer(monkeypatch, values, True)
    with pytest.raises(ValueError):
        method._native_staging.copy(layer, 6, 0)
    with pytest.raises(ValueError):
        method._native_staging.copy(layer, 0, 3)

    class ForeignRows(CompressedGGUFRows):
        pass

    foreign = ForeignRows.__new__(ForeignRows)
    foreign.__dict__.update(values[0].__dict__)
    assert make_expert_copies([foreign, *values[1:]], layer, 6) is None
    monkeypatch.setenv("SGLANG_XPU_GGUF_NATIVE_STAGE", "0")
    assert make_expert_copies(values, layer, 6) is None


def test_native_group_order_queued_lifetimes_and_rejections(monkeypatch):
    values = sources(22)
    method, layer = method_layer(monkeypatch, values, True)
    snapshots = []
    for repeat in range(72):
        rows = (1, 2, 4)[repeat % 3]
        topk = 1 if repeat % 4 == 0 else 10
        ids = np.asarray([0, 2, 4], dtype=np.int32)[
            (np.arange(rows * topk) + repeat) % 3
        ].reshape(rows, topk)
        resident = {
            int(expert): (index + repeat) % 3
            for index, expert in enumerate(np.unique(ids))
        }
        actual = method._native_staging.group_routes(ids, resident)
        expected = group_iq2s_routes(ids, resident, "cpu")
        snapshots.append((actual, expected))
        if repeat % 8 == 0:
            # Mix the small route image with full expert transfers in the ring.
            method._ensure_expert(layer, repeat % 6)
    for actual, expected in snapshots:
        assert actual.groups == expected.groups
        for name in (
            "data",
            "route_slots",
            "group_slots",
            "starts",
            "token_rows",
            "destinations",
        ):
            assert torch.equal(getattr(actual, name).cpu(), getattr(expected, name))
    with pytest.raises(ValueError):
        method._native_staging.group_routes(np.array([[1]], dtype=np.int32), {0: 0})
    with pytest.raises(ValueError):
        method._native_staging.group_routes(
            np.array([[0, 2]], dtype=np.int32), {0: 0, 2: 0}
        )
