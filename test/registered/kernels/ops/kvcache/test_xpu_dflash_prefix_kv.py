"""Dynamic prefix writes must preserve all uncommitted cache bytes."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU required")
class TestXpuDFlashPrefixKV(unittest.TestCase):
    def pool(self, dtype=torch.bfloat16, width=1024, v_width=None):
        v_width = width if v_width is None else v_width
        pool = SimpleNamespace(
            dtype=dtype,
            store_dtype=dtype,
            row_dim=width,
            v_row_dim=v_width,
            start_layer=0,
            k_buffer=[torch.zeros(256, 8, width // 8, device="xpu", dtype=dtype)],
            v_buffer=[torch.zeros(256, 8, v_width // 8, device="xpu", dtype=dtype)],
            fallbacks=0,
        )

        def store(layer, loc, k, v, *args, **kwargs):
            pool.fallbacks += 1
            pool.k_buffer[0].index_copy_(0, loc, k)
            pool.v_buffer[0].index_copy_(0, loc, v)

        pool.set_kv_buffer = store
        return pool

    def invoke(self, pool, loc, lens, k, v):
        MHATokenToKVPool.set_kv_buffer_prefix_valid(
            pool, SimpleNamespace(layer_id=0), loc, lens, k, v
        )

    def test_graph_dynamic_commit_locations_and_strided_source(self):
        torch.manual_seed(42)
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
            envs.SGLANG_XPU_DFLASH_PREFIX_KV.override(True),
        ):
            for bs in (1, 3):
                for n in range(2, 9):
                    with self.subTest(bs=bs, n=n):
                        pool = self.pool()
                        # Layer views from a stacked context projection need
                        # not be contiguous in the token dimension.
                        raw_k = torch.randn(
                            bs * n, 2, 8, 128, device="xpu", dtype=torch.bfloat16
                        )
                        raw_v = torch.randn_like(raw_k)
                        k, v = raw_k[:, 0], raw_v[:, 1]
                        loc = torch.arange(
                            bs * n, device="xpu", dtype=torch.int64
                        ).view(bs, n)
                        lens = torch.full((bs,), n, device="xpu", dtype=torch.int32)
                        self.invoke(pool, loc, lens, k, v)
                        torch.xpu.synchronize()
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph):
                            self.invoke(pool, loc, lens, k, v)
                        for step in range(n + 1):
                            raw_k.normal_()
                            raw_v.normal_()
                            lengths = [(step + b) % (n + 1) for b in range(bs)]
                            lens.copy_(
                                torch.tensor(lengths, device="xpu", dtype=torch.int32)
                            )
                            ids = (
                                torch.arange(bs * n, device="xpu", dtype=torch.int64)
                                + 19 * step
                            ).view(bs, n)
                            expected_k = torch.full_like(pool.k_buffer[0], -3)
                            expected_v = torch.full_like(pool.v_buffer[0], 7)
                            pool.k_buffer[0].copy_(expected_k)
                            pool.v_buffer[0].copy_(expected_v)
                            for b, length in enumerate(lengths):
                                expected_k[ids[b, :length]] = k[b * n : b * n + length]
                                expected_v[ids[b, :length]] = v[b * n : b * n + length]
                                # Uncommitted positions must never be read.
                                ids[b, length:] = 1000000
                            loc.copy_(ids)
                            graph.replay()
                            torch.testing.assert_close(
                                pool.k_buffer[0], expected_k, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                pool.v_buffer[0], expected_v, rtol=0, atol=0
                            )
                        self.assertEqual(pool.fallbacks, 0)
                        del graph

    def test_guards_keep_existing_fallback(self):
        with envs.SGLANG_XPU_PAGED_DFLASH.override(True):
            for enabled, dtype, width, v_width in (
                (False, torch.bfloat16, 1024, 1024),
                (True, torch.float16, 1024, 1024),
                (True, torch.bfloat16, 1024, 512),
            ):
                with envs.SGLANG_XPU_DFLASH_PREFIX_KV.override(enabled):
                    pool = self.pool(dtype, width, v_width)
                    k = torch.randn(4, 8, width // 8, device="xpu", dtype=dtype)
                    v = torch.randn(4, 8, v_width // 8, device="xpu", dtype=dtype)
                    loc = torch.tensor([[7, 8, 9, 10]], device="xpu", dtype=torch.int64)
                    lens = torch.tensor([2], device="xpu", dtype=torch.int32)
                    self.invoke(pool, loc, lens, k, v)
                    self.assertEqual(pool.fallbacks, 1)
                    torch.testing.assert_close(
                        pool.k_buffer[0][7:9], k[:2], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        pool.v_buffer[0][7:9], v[:2], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
