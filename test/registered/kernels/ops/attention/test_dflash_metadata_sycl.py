"""Single-request SYCL metadata against the retained attention backend."""

import itertools
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dflash_metadata_sycl import (
    OUTPUT_NAMES,
    try_update_metadata,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.mem_cache.kv_index_translator import KVIndexTable, KVIndexTranslator
from sglang.srt.mem_cache.kv_loc_plan import IdSpace, IdSpaceKind
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=45, suite="xpu-unit-test")


WINDOW = int(os.environ.get("SGLANG_TEST_DFLASH_NATIVE_WINDOW", "4"))


@unittest.skipUnless(
    torch.xpu.is_available() and os.environ.get("SGLANG_XPU_DFLASH_NATIVE_LIB"),
    "requires B70 and an explicitly built native library",
)
class TestDflashMetadataSycl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("B70 metadata")

    def fixture(
        self,
        kind="tokens",
        idtype=torch.int32,
        rowtype=torch.int64,
        seqtype=torch.int64,
        indextype=torch.int64,
    ):
        seq = torch.zeros(1, device="xpu", dtype=seqtype)
        req = torch.tensor([2], device="xpu", dtype=rowtype)
        v2p = torch.randint(-1, 192, (256,), device="xpu", dtype=torch.int64)
        pages = torch.randint(0, 256, (4, 64), device="xpu", dtype=idtype)
        ids = (
            pages
            if kind == "pages"
            else (pages[:, :, None] * 128 + torch.arange(128, device="xpu"))
            .flatten(1)
            .to(idtype)
        )
        if kind != "pages":
            ids[2, 256:384] = -1
        translator = object.__new__(KVIndexTranslator)
        translator.page_size = 128
        translator.req_to_token = ids
        translator.defer_read_translate = False
        translator._rows = torch.arange(4, device="xpu", dtype=rowtype)
        translator._spaces = {
            IdSpaceKind.FULL: IdSpace(
                key="metadata-test", read_v2p=v2p if kind != "tokens" else None
            )
        }
        table = KVIndexTable(
            ids=ids,
            row_ids=req,
            row_stride=ids.stride(0),
            entry_page_size=128,
            is_translated=True,
        )
        plan = SimpleNamespace(
            has_read_table=lambda *args, **kwargs: kind == "pages",
            is_read_by=lambda *args: True,
            read_table=lambda **kwargs: table,
        )
        backend = object.__new__(TritonAttnBackend)
        backend.device = "xpu"
        backend.is_draft_runner = True
        backend.dcp_size = 1
        backend.max_context_len = 4096
        backend.sliding_window_size = 2047
        backend.token_to_kv_pool = None
        backend._verify_mask = None
        backend.kv_index_translator = translator
        for name in OUTPUT_NAMES:
            if name in ("cuda_graph_kv_indices", "cuda_graph_window_kv_indices"):
                value = torch.full((8192,), -77, device="xpu", dtype=torch.int64)
            elif name == "cuda_graph_window_kv_offsets":
                value = torch.full((1,), -77, device="xpu", dtype=torch.int32)
            else:
                dtype = (
                    indextype if name in ("qo_indptr", "mask_indptr") else torch.int32
                )
                value = torch.tensor([0, -77], device="xpu", dtype=dtype)
            setattr(backend, name, value)
        backend.cuda_graph_window_num_kv_splits = torch.zeros(
            1, device="xpu", dtype=torch.int32
        )
        spec = SimpleNamespace(num_tokens_per_req=WINDOW, custom_mask=None)
        return backend, plan, seq, spec, req, v2p

    def test_reference_and_live_graph_sources(self):
        torch.manual_seed(42)
        types = (torch.int32, torch.int64)
        for kind, idtype, rowtype, seqtype, indextype in itertools.product(
            ("tokens", "virtual", "pages"), types, types, types, types
        ):
            b, plan, seq, spec, req, v2p = self.fixture(
                kind, idtype, rowtype, seqtype, indextype
            )
            self.assertIsNotNone(try_update_metadata(b, plan, 1, seq, spec, req))
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                try_update_metadata(b, plan, 1, seq, spec, req)
            for length in (0, 1, 127, 128, 129, 1024, 2047, 2048, 4096):
                seq[0] = length
                v2p.random_(-1, 192)
                before = [getattr(b, n).clone() for n in OUTPUT_NAMES]
                with envs.SGLANG_XPU_DFLASH_SYCL_METADATA.override(False):
                    b._update_target_verify_buffers(plan, 1, seq, spec, req)
                expected = [getattr(b, n).clone() for n in OUTPUT_NAMES]
                for name, value in zip(OUTPUT_NAMES, before):
                    getattr(b, name).copy_(value)
                graph.replay()
                for name, value in zip(OUTPUT_NAMES, expected):
                    torch.testing.assert_close(
                        getattr(b, name), value, rtol=0, atol=0, msg=name
                    )
            del graph

    def test_unsupported_contract_leaves_outputs_untouched(self):
        for change in (
            "width",
            "batch",
            "target",
            "dcp",
            "window",
            "mask",
            "capacity",
            "dtype",
        ):
            b, plan, seq, spec, req, _ = self.fixture()
            bs = 1
            if change == "width":
                spec.num_tokens_per_req = 3
            elif change == "batch":
                bs = 2
            elif change == "target":
                b.is_draft_runner = False
            elif change == "dcp":
                b.dcp_size = 2
            elif change == "window":
                b.sliding_window_size = 2048
            elif change == "mask":
                spec.custom_mask = torch.ones(1, device="xpu")
            elif change == "capacity":
                b.cuda_graph_kv_indices = b.cuda_graph_kv_indices[:1]
            else:
                b.qo_indptr = b.qo_indptr.float()
            before = [getattr(b, n).clone() for n in OUTPUT_NAMES]
            self.assertIsNone(try_update_metadata(b, plan, bs, seq, spec, req))
            for name, value in zip(OUTPUT_NAMES, before):
                torch.testing.assert_close(getattr(b, name), value, rtol=0, atol=0)

    def test_master_child_dispatch(self):
        target = "sglang.kernels.ops.attention.dflash_metadata_sycl.try_update_metadata"
        for master, child in ((False, True), (True, False)):
            b, plan, seq, spec, req, _ = self.fixture()
            with (
                envs.SGLANG_XPU_PAGED_DFLASH.override(master),
                envs.SGLANG_XPU_DFLASH_SYCL_METADATA.override(child),
                patch(target, side_effect=AssertionError("retained metadata expected")),
            ):
                b._update_target_verify_buffers(plan, 1, seq, spec, req)
        b, plan, seq, spec, req, _ = self.fixture()
        with (
            envs.SGLANG_XPU_PAGED_DFLASH.override(True),
            envs.SGLANG_XPU_DFLASH_SYCL_METADATA.override(True),
            patch(target, return_value=("native",)) as native,
        ):
            self.assertEqual(
                b._update_target_verify_buffers(plan, 1, seq, spec, req), ("native",)
            )
            native.assert_called_once()
