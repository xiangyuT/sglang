"""Native paging parity under changing routes, misses and graph replay."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.xpu_compact_moe import (
    prepare_routes,
    sort_routes,
    xpu_compact_moe,
)
from sglang.kernels.ops.moe.xpu_compact_paged import _PagedMatmul
from sglang.srt.environ import envs


@unittest.skipUnless(
    torch.xpu.is_available() and os.environ.get("SGLANG_XPU_DFLASH_NATIVE_LIB"),
    "XPU and explicitly built native paged MoE library required",
)
class TestDflashFusedRoutes(unittest.TestCase):
    sycl_backend = False
    window = 4

    def bind_candidate(self, matmul, *args):
        matmul.bind_fused_routes(*args)

    @classmethod
    def setUpClass(cls):
        import sgl_kernel  # noqa: F401 -- registers the native Silu/combine ops

        torch.manual_seed(42)
        cls.hosts = []
        for shape, dtype in (
            ((256, 1024, 2048), torch.float8_e4m3fn),
            ((256, 2048, 512), torch.float8_e4m3fn),
            ((256, 8, 16), torch.float32),
            ((256, 16, 4), torch.float32),
        ):
            host = torch.empty(shape, dtype=dtype, pin_memory=True)
            if dtype == torch.float32:
                host.uniform_(0.0001, 0.002)
            else:
                for start in range(0, 256, 8):
                    host[start : start + 8].copy_(
                        (torch.randn(8, *shape[1:]) * 16).to(dtype)
                    )
            cls.hosts.append(host)
        cls.caches = [[h[:192].to("xpu") for h in cls.hosts] for _ in range(2)]
        cls.hp = torch.tensor(
            [h.data_ptr() for h in cls.hosts], device="xpu", dtype=torch.uint64
        )
        cls.gps = [
            torch.tensor(
                [g.data_ptr() for g in cache], device="xpu", dtype=torch.uint64
            )
            for cache in cls.caches
        ]
        cls.state = SimpleNamespace(
            src=torch.zeros(192, device="xpu", dtype=torch.int32),
            dst=torch.zeros(192, device="xpu", dtype=torch.int32),
            count=torch.zeros(1, device="xpu", dtype=torch.int32),
        )

    def paging(self, index):
        return self.hp, self.gps[index], self.state, self.hosts

    def reset_caches(self):
        for cache in self.caches:
            for gpu, host in zip(cache, self.hosts):
                gpu.copy_(host[:192], non_blocking=True)

    def check_caches(self):
        for old, new in zip(*self.caches):
            self.assertTrue(torch.equal(old.view(torch.uint8), new.view(torch.uint8)))

    def cases(self):
        # Include slot 0 and 191, shared rows, repeated IDs, and 32 rows for a
        # single expert (multiple native M tiles). Every miss is actively routed.
        n = self.window * 8
        for shift in (0, 192 - n):
            routes = torch.arange(self.window * 8, dtype=torch.int32) + shift
            for misses in sorted({0, 1, 2, 4, 8, 20, 32, n}):
                yield routes.view(self.window, 8), misses
        yield (
            (torch.arange(self.window * 8, dtype=torch.int32) % 8).view(self.window, 8),
            8,
        )
        yield (
            (torch.arange(self.window * 8, dtype=torch.int32) % 20).view(
                self.window, 8
            ),
            4,
        )
        yield torch.full((self.window, 8), 191, dtype=torch.int32), 1
        yield torch.zeros(self.window, 8, dtype=torch.int32), 1

    def set_plan(self, routes, misses, step):
        destinations = routes.flatten().unique()[:misses]
        sources = (torch.arange(misses, dtype=torch.int32) * 37 + step * 11) % 256
        self.state.src.fill_(-1)
        self.state.dst.fill_(-1)
        self.state.src[:misses].copy_(sources)
        self.state.dst[:misses].copy_(destinations)
        self.state.count.fill_(misses)

    def test_route_metadata_and_scale_cache_replay(self):
        top = torch.arange(self.window * 8, device="xpu", dtype=torch.int32).view(
            self.window, 8
        )
        self.state.count.zero_()

        def metadata(index):
            sorted_routes, cmap, ids, counts, offsets = [
                torch.empty(self.window * 8, device="xpu", dtype=torch.int32)
                for _ in range(5)
            ]
            matmul = _PagedMatmul(self.paging(index))
            if index:
                self.bind_candidate(
                    matmul, top, sorted_routes, cmap, ids, counts, offsets, 192
                )
            else:
                slots = torch.empty_like(ids)
                sort_routes[(1,)](
                    top,
                    sorted_routes,
                    slots,
                    cmap,
                    R=self.window * 8,
                    BR=self.window * 8,
                    E=192,
                    num_warps=4,
                )
                prepare_routes[(1,)](
                    slots,
                    ids,
                    counts,
                    offsets,
                    R=self.window * 8,
                    BR=self.window * 8,
                    num_warps=4,
                )
                matmul.bind_routes(ids)
            return sorted_routes, cmap, ids, counts, offsets, matmul.host_ids

        metadata(0)
        metadata(1)
        torch.xpu.synchronize()
        graphs, outputs = [], []
        for index in range(2):
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                outputs.append(metadata(index))
            graphs.append(graph)
        for step, (routes, misses) in enumerate(self.cases()):
            with self.subTest(step=step, misses=misses):
                top.copy_(routes)
                self.set_plan(routes, misses, step)
                self.reset_caches()
                for graph in graphs:
                    graph.replay()
                for old, new in zip(*outputs):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
                self.check_caches()

    def run_moe(
        self, index, x, ids, weights, *, master=True, enabled=True, paging=True
    ):
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(master),
            envs.SGLANG_XPU_DFLASH_FUSED_ROUTES.override(enabled),
            envs.SGLANG_XPU_DFLASH_SYCL_ROUTES.override(self.sycl_backend and enabled),
        ):
            return xpu_compact_moe(
                x,
                *self.caches[index],
                ids,
                weights,
                paging=self.paging(index) if paging else None,
            )

    def test_full_moe_eager_and_changing_graph_inputs(self):
        x = torch.randn(self.window, 2048, device="xpu", dtype=torch.bfloat16)
        ids = torch.arange(self.window * 8, device="xpu", dtype=torch.int32).view(
            self.window, 8
        )
        weights = torch.full((self.window, 8), 0.125, device="xpu")
        self.state.count.zero_()
        for index in range(2):
            self.run_moe(index, x, ids, weights, enabled=bool(index))
        torch.xpu.synchronize()
        graphs, outputs = [], []
        for index in range(2):
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                outputs.append(
                    self.run_moe(index, x, ids, weights, enabled=bool(index))
                )
            graphs.append(graph)
        for step, (routes, misses) in enumerate(self.cases()):
            with self.subTest(step=step, misses=misses):
                x.normal_()
                weights.uniform_()
                weights /= weights.sum(-1, keepdim=True)
                if step == 16:
                    x.zero_()
                if step == 17:
                    weights.zero_()
                ids.copy_(routes)
                self.set_plan(routes, misses, step)
                self.reset_caches()
                eager = [
                    self.run_moe(index, x, ids, weights, enabled=bool(index))
                    for index in range(2)
                ]
                torch.testing.assert_close(*eager, rtol=0, atol=0)
                self.check_caches()
                self.reset_caches()
                for _ in range(2):
                    for graph in graphs:
                        graph.replay()
                    for output in outputs:
                        torch.testing.assert_close(output, eager[0], rtol=0, atol=0)
                    self.check_caches()

    def test_master_child_shape_and_paging_guards(self):
        self.state.count.zero_()
        self.reset_caches()
        target = "sglang.kernels.ops.moe.xpu_compact_paged._PagedMatmul." + (
            "bind_sycl_routes" if self.sycl_backend else "bind_fused_routes"
        )
        with patch(target, side_effect=AssertionError("must use original route path")):
            for m, master, enabled, paging in (
                (4, False, True, True),
                (4, True, False, True),
                (4, True, True, False),
                (2, True, True, True),
                (3, True, True, True),
                (5, True, True, True),
                (8, True, True, True),
            ):
                if self.sycl_backend and m == 8 and master and enabled and paging:
                    continue  # M8 is now explicitly supported by the SYCL child.
                with self.subTest(m=m, master=master, enabled=enabled, paging=paging):
                    x = torch.randn(m, 2048, device="xpu", dtype=torch.bfloat16)
                    ids = torch.arange(m * 8, device="xpu", dtype=torch.int32).view(
                        m, 8
                    )
                    weights = torch.full((m, 8), 0.125, device="xpu")
                    actual = self.run_moe(
                        1,
                        x,
                        ids,
                        weights,
                        master=master,
                        enabled=enabled,
                        paging=paging,
                    )
                    expected = self.run_moe(
                        0, x, ids, weights, enabled=False, paging=paging
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
