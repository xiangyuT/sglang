"""Fused submission must preserve the existing Q8 expert pipeline math."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
import torch.nn.functional as F

from sglang.kernels.ops.quantization.gguf_moe_q8 import fused_q8_experts
from sglang.kernels.ops.quantization.gguf_sycl import _library, expert_matvec_gguf
from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=15, suite="xpu-unit-test")


def packed_cache(kind, count, capacity, seed):
    block, size = BLOCK_GEOMETRY[GGML_TYPES[kind]]
    raw = np.random.default_rng(seed).integers(
        0, 256, (count // block, size), dtype=np.uint8
    )
    if kind == 29:
        scales = raw[:, 48:].copy().view(np.uint16)
        for i in range(4):
            scales[:, i] = (scales[:, i] & 0xFFF) | (((0x3400 >> (4 * i)) & 15) << 12)
        raw[:, 48:] = scales.view(np.uint8)
    else:
        raw[:, :2] = np.array([0.25], dtype=np.float16).view(np.uint8)
    return torch.from_numpy(raw).view(capacity, -1).to("xpu")


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestFusedGGUFQ8MoE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 kernel")
        path = os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB")
        if not path or not hasattr(_library(path), "xpu_gguf_moe_q8"):
            raise unittest.SkipTest("Build the fused GGUF library")

    def test_mixed_packet_graph_paging_preserves_existing_fp32(self):
        from sglang.kernels.ops.quantization.gguf_gpu_cache import GPUExpertCache

        if not hasattr(_library(os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB")),
                       "xpu_gguf_mixed_packets"):
            self.skipTest("Build the mixed expert packet library")
        flag = "SGLANG_XPU_GGUF_MIXED_PACKETS"
        for kind, down_kind in ((21, 20), (21, 8), (23, 8)):
            raw = [packed_cache(k, 9 * 640 * 2560, 9, 3500 + i)
                   for i, k in enumerate((kind, kind, down_kind))]
            for tensor in raw:
                tensor[8].zero_()
            buffers = [value[:4].clone() for value in raw]
            mirrors = [value[4:].cpu().pin_memory() for value in raw]
            cache = GPUExpertCache(kind, range(4), buffers, range(4, 9), mirrors,
                                   layout="original", down_kind=down_kind)
            try:
                for count in (1, 4):
                    with self.subTest(kind=kind, down_kind=down_kind, tokens=count):
                        ids = torch.zeros((count, 10), dtype=torch.int32, device="xpu")
                        weights = torch.full((count, 10), .1, device="xpu")
                        x = torch.zeros((count, 2560), dtype=torch.bfloat16, device="xpu")
                        spaces = {}

                        def forward():
                            prepared = cache.prepare(ids)
                            return fused_q8_experts(
                                x, *buffers, kind, prepared.slots, weights, spaces,
                                grouping=prepared.grouping,
                                slot_addresses=cache.slot_addresses, down_kind=down_kind,
                            )

                        def workspace(output):
                            return next(v for v in spaces.values() if v["output"] is output)

                        with patch.dict(os.environ, {flag: "1"}):
                            forward()
                            torch.xpu.synchronize()
                            graph = torch.xpu.XPUGraph()
                            with torch.xpu.graph(graph):
                                captured = forward()
                        for seed, pair in ((3, (0, 1)), (7, (4, 5)), (11, (8, 8))):
                            generator = torch.Generator().manual_seed(seed)
                            x.copy_(torch.randn(count, 2560, generator=generator) * .01)
                            ids.copy_(torch.tensor([[*pair] * 4 + [8, 8]] * count,
                                                   dtype=torch.int32))
                            with patch.dict(os.environ, {flag: "0"}):
                                original = forward()
                                expected = {key: value.clone() for key, value in workspace(original).items()
                                            if key in ("gate", "up", "activation", "down", "output")}
                            with patch.dict(os.environ, {flag: "1"}):
                                eager = forward()
                            graph.replay()
                            for output in (eager, captured):
                                for key, reference in expected.items():
                                    actual = workspace(output)[key]
                                    dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
                                    self.assertTrue(torch.equal(actual.cpu().view(dtype),
                                                                reference.cpu().view(dtype)), key)
                        del graph
            finally:
                cache.close()

    def test_intermediates_and_dynamic_graph_against_composed_q8(self):
        generator = torch.Generator().manual_seed(42)
        for kind in (16, 22, 29, 42):
            gate = packed_cache(kind, 4 * 640 * 2560, 4, kind)
            up = packed_cache(kind, 4 * 640 * 2560, 4, kind + 42)
            down = packed_cache(42, 4 * 2560 * 640, 4, 83)
            for count, topk in ((1, 1), (1, 10), (4, 10)):
                with (
                    self.subTest(kind=kind, count=count, topk=topk),
                    torch.inference_mode(),
                ):
                    routes = count * topk
                    x = (torch.randn(count, 2560, generator=generator) * 0.01).to(
                        device="xpu", dtype=torch.bfloat16
                    )
                    slots = (
                        torch.arange(routes, device="xpu", dtype=torch.int32) % 4
                    ).contiguous()
                    weights = torch.full(
                        (count, topk),
                        1 / topk,
                        device="xpu",
                        dtype=torch.bfloat16 if kind == 42 else torch.float32,
                    )
                    token_rows = torch.arange(
                        count, device="xpu", dtype=torch.int32
                    ).repeat_interleave(topk)
                    workspaces = {}

                    def reference():
                        g, u = expert_matvec_gguf(
                            x.float(),
                            gate,
                            kind,
                            slots,
                            640,
                            up=up,
                            tokens=token_rows,
                            q8_activations=True,
                        )
                        activation = F.silu(g) * u
                        y = expert_matvec_gguf(
                            activation, down, 42, slots, 2560, q8_activations=True
                        )
                        output = (
                            (y.view(count, topk, 2560) * weights.float().unsqueeze(-1))
                            .sum(1)
                            .bfloat16()
                        )
                        return g, u, activation, y, output

                    def forward():
                        return fused_q8_experts(
                            x, gate, up, down, kind, slots, weights, workspaces
                        )

                    def check(output):
                        g, u, activation, y, expected = reference()
                        ws = next(
                            value
                            for value in workspaces.values()
                            if value["output"].data_ptr() == output.data_ptr()
                        )
                        self.assertTrue(
                            torch.equal(
                                ws["gate"].cpu().view(torch.int32),
                                g.cpu().view(torch.int32),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                ws["up"].cpu().view(torch.int32),
                                u.cpu().view(torch.int32),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                ws["activation"].cpu().view(torch.int32),
                                activation.cpu().view(torch.int32),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                ws["down"].cpu().view(torch.int32),
                                y.cpu().view(torch.int32),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                output.cpu().view(torch.int16),
                                expected.cpu().view(torch.int16),
                            )
                        )

                    for _ in range(3):
                        forward()
                    check(forward())
                    graph = torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph):
                        output = forward()
                    for scale in (0.5, -2.0, 1.0):
                        x.mul_(scale)
                        slots.copy_(torch.roll(slots, 1))
                        graph.replay()
                        check(output)

    @torch.inference_mode()
    def test_resident_mtp_batch_waves_replay_updated_inputs_without_host_routes(self):
        for count in (12, 512):
            with self.subTest(experts=count):
                self._check_resident_mtp_batch_waves(count)

    def _check_resident_mtp_batch_waves(self, count):
        from types import SimpleNamespace

        from sglang.kernels.ops.quantization.gguf_gpu_cache import GPUExpertCache
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.quantization.qwen4_gguf_xpu import (
            CompressedGGUFMoEMethod,
        )

        buffers = tuple(
            packed_cache(42, count * 640 * 2560, count, 337 + i) for i in range(3)
        )
        mirrors = tuple(
            torch.empty((0, t.shape[1]), dtype=torch.uint8).pin_memory()
            for t in buffers
        )
        cache = GPUExpertCache(
            42, range(count), buffers, (), mirrors, layout="original"
        )
        method = CompressedGGUFMoEMethod(count, decode_q8=True)
        method.num_experts = count
        method.hidden_size, method.intermediate = 2560, 640
        method.sources = (SimpleNamespace(kind=42),) * 3
        method._gpu_cache = cache
        method.fused_q8 = True
        layer = SimpleNamespace(
            gguf_gate=buffers[0], gguf_up=buffers[1], gguf_down=buffers[2]
        )
        mode = ForwardMode.DRAFT_EXTEND_V2
        try:
            for rows in (8, 32):
                x = torch.zeros((rows, 2560), dtype=torch.bfloat16, device="xpu")
                ids = torch.zeros((rows, 10), dtype=torch.int32, device="xpu")
                weights = torch.full((rows, 10), 0.1, dtype=torch.float32, device="xpu")
                batch = StandardDispatchOutput(
                    hidden_states=x,
                    hidden_states_scale=None,
                    topk_output=StandardTopKOutput(weights, ids, None),
                )
                with patch(
                    "sglang.srt.model_executor.runner_utils.forward_batch.get_forward_batch",
                    return_value=SimpleNamespace(forward_mode=mode),
                ):
                    sentinel = torch.ones_like(x)
                    with (
                        patch.dict(
                            os.environ, {"SGLANG_XPU_GGUF_MTP_BATCH_WAVES": "0"}
                        ),
                        patch.object(
                            method, "_apply_routes", return_value=sentinel
                        ) as fallback,
                    ):
                        self.assertIs(
                            method.apply(layer, batch).hidden_states, sentinel
                        )
                        fallback.assert_called_once()
                    with (
                        patch.dict(
                            os.environ, {"SGLANG_XPU_GGUF_MTP_BATCH_WAVES": "1"}
                        ),
                        patch.object(
                            method,
                            "_apply_routes",
                            side_effect=AssertionError("host routing reached"),
                        ),
                    ):
                        method.apply(layer, batch)
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph):
                            result = method.apply(layer, batch).hidden_states
                        for seed in (11, 19, 43):
                            g = torch.Generator().manual_seed(seed)
                            x.copy_(torch.randn(rows, 2560, generator=g) * 0.01)
                            ids.copy_(
                                torch.randint(
                                    count, (rows, 10), generator=g, dtype=torch.int32
                                )
                            )
                            weights.copy_(
                                torch.softmax(
                                    torch.randn(rows, 10, generator=g), dim=-1
                                )
                            )
                            expected = []
                            for first in range(0, rows, 4):
                                wave = StandardDispatchOutput(
                                    hidden_states=x[first : first + 4],
                                    hidden_states_scale=None,
                                    topk_output=StandardTopKOutput(
                                        weights[first : first + 4],
                                        ids[first : first + 4],
                                        None,
                                    ),
                                )
                                expected.append(
                                    method.apply(layer, wave).hidden_states.clone()
                                )
                            graph.replay()
                            self.assertTrue(
                                torch.equal(
                                    result.cpu().view(torch.int16),
                                    torch.cat(expected).cpu().view(torch.int16),
                                )
                            )
                        del graph
        finally:
            cache.close()


if __name__ == "__main__":
    unittest.main()
