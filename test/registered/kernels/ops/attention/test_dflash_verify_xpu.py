import unittest

import torch

from sglang.kernels.ops.attention.dflash_verify_xpu import (
    make_dflash_verify_decode_metadata,
)


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestDFlashVerifyXpu(unittest.TestCase):
    def test_decode_equivalence_and_dynamic_graph_metadata(self):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        torch.manual_seed(42)
        for m in range(2, 9):
            q = torch.randn(m, 16, 256, device="xpu", dtype=torch.bfloat16)
            k = torch.randn(64, 128, 2, 256, device="xpu", dtype=q.dtype)
            v = torch.randn_like(k)
            table = torch.randperm(64, device="xpu", dtype=torch.int32)[:32][None]
            lengths = torch.tensor([m], device="xpu", dtype=torch.int32)
            out = torch.empty_like(q)

            def metadata(**overrides):
                args = dict(
                    q=q,
                    key_cache=k,
                    value_cache=v,
                    page_table=table,
                    cache_seqlens=lengths,
                    max_seqlen_q=m,
                )
                args.update(overrides)
                return make_dflash_verify_decode_metadata(**args)

            def op(splits):
                pages, lens, cuq = metadata()
                return flash_attn_with_kvcache(
                    q=q,
                    k_cache=k,
                    v_cache=v,
                    page_table=pages,
                    cache_seqlens=lens,
                    cu_seqlens_q=cuq,
                    max_seqlen_q=1,
                    softmax_scale=256**-0.5,
                    causal=False,
                    num_splits=splits,
                    out=out,
                )

            for splits in (0, 1):
                op(splits)
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    op(splits)
                for prefix in (0, 127, 128, 1024, 4096 - m):
                    with self.subTest(tokens=m, prefix=prefix, splits=splits):
                        q.normal_()
                        lengths.fill_(prefix + m)
                        table.copy_(
                            torch.randperm(64, device="xpu", dtype=torch.int32)[:32][
                                None
                            ]
                        )
                        reference = []
                        for i in range(m):
                            reference.append(
                                flash_attn_with_kvcache(
                                    q=q[i : i + 1],
                                    k_cache=k,
                                    v_cache=v,
                                    page_table=table,
                                    cache_seqlens=lengths - m + i + 1,
                                    cu_seqlens_q=torch.tensor(
                                        [0, 1], device="xpu", dtype=torch.int32
                                    ),
                                    max_seqlen_q=1,
                                    softmax_scale=256**-0.5,
                                    causal=False,
                                    num_splits=splits,
                                )
                            )
                        graph.replay()
                        torch.testing.assert_close(
                            out, torch.cat(reference), rtol=0, atol=0
                        )
                del graph

            # Unsupported layouts return without changing any input or output.
            self.assertIsNone(metadata(page_table=table.expand(2, -1)))
            self.assertIsNone(metadata(max_seqlen_q=1))
            self.assertIsNone(metadata(q=q[:, :8]))
            self.assertIsNone(metadata(key_cache=k[:, :64]))
            self.assertIsNone(metadata(cache_seqlens=lengths.long()))


if __name__ == "__main__":
    unittest.main()
