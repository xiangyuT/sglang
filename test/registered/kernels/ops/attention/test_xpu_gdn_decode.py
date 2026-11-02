"""B70 indexed GDN decode versus native and independent FP64 recurrence."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from sgl_kernel import gdn_attention
from sglang.kernels.ops.attention.xpu_gdn_decode import (
    can_use_xpu_gdn_decode,
    xpu_gdn_decode,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    scatter_mamba_states_after_mtp_verify,
)
from sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=45, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuGdnDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.xpu.get_device_properties(0).device_id != 0xE223:
            raise unittest.SkipTest("B70 native rounding contract")

    def _inputs(self):
        torch.manual_seed(42)
        q = torch.randn(1, 12288, device="xpu", dtype=torch.bfloat16) * 0.2
        ba = torch.randn(1, 64, device="xpu", dtype=torch.bfloat16)
        cs = torch.randn(5, 8192, 3, device="xpu", dtype=torch.bfloat16) * 0.1
        ss = torch.randn(5, 32, 128, 128, device="xpu") * 0.1
        w = torch.randn(8192, 4, device="xpu", dtype=torch.bfloat16) * 0.05
        alog = torch.rand(32, device="xpu") * 4 - 2
        bias = torch.rand(32, device="xpu", dtype=torch.bfloat16)
        idx = torch.tensor([2], device="xpu", dtype=torch.int64)
        return q, ba, cs, ss, w, alog, bias, idx

    def _native(self, args):
        q, ba, cs, ss, w, alog, bias, idx = args
        out = torch.empty((1, 32, 128), device="xpu", dtype=q.dtype)
        z = torch.empty_like(out)
        gdn_attention(
            core_attn_out=out,
            z=z,
            projected_states_qkvz=q,
            projected_states_ba=ba,
            num_k_heads=16,
            num_v_heads=32,
            head_k_dim=128,
            head_v_dim=128,
            conv_state=cs.transpose(1, 2),
            ssm_state=ss,
            conv_weights=w,
            conv_bias=None,
            activation="silu",
            A_log=alog,
            dt_bias=bias,
            num_prefills=0,
            num_decodes=1,
            num_spec_decodes=0,
            has_initial_state=torch.ones(1, device="xpu", dtype=torch.bool),
            non_spec_query_start_loc=torch.tensor(
                [0, 1], device="xpu", dtype=torch.int32
            ),
            non_spec_token_indx=None,
            non_spec_state_indices_tensor=idx.int(),
            spec_query_start_loc=None,
            spec_token_indx=None,
            spec_state_indices_tensor=None,
            num_accepted_tokens=None,
            num_actual_tokens=1,
            tp_size=1,
            reorder_input=True,
        )
        return out, z

    def test_block_verify_and_commit_every_prefix(self):
        for steps in range(2, 9):
            for captured in (False, True):
                with self.subTest(steps=steps, captured=captured):
                    self._check_verify_prefixes(steps, captured)

    def _check_verify_prefixes(self, steps, captured):
        _, _, cs0, ss0, weight, alog, bias, indices = self._inputs()
        qkvz = torch.randn(steps, 12288, device="xpu", dtype=torch.bfloat16) * 0.2
        ba = torch.randn(steps, 64, device="xpu", dtype=torch.bfloat16)
        cs_ref, ss_ref = cs0.clone(), ss0.clone()
        ref_out, ref_conv, ref_state = [], [], []
        for token in range(steps):
            out, _ = self._native(
                (
                    qkvz[token : token + 1],
                    ba[token : token + 1],
                    cs_ref,
                    ss_ref,
                    weight,
                    alog,
                    bias,
                    indices,
                )
            )
            ref_out.append(out.clone())
            ref_conv.append(cs_ref[2].clone())
            ref_state.append(ss_ref[2].clone())
        cs, ss = cs0.clone(), ss0.clone()
        conv_steps = torch.empty(2, steps, 8192, 3, device="xpu", dtype=cs.dtype)
        ssm_steps = torch.empty(2, steps, 32, 128, 128, device="xpu")
        intermediate_indices = torch.tensor([1], device="xpu", dtype=torch.int64)
        cu_seqlens = torch.tensor([0, steps], device="xpu", dtype=torch.int32)
        # Exercise DFlash's chain metadata, including the generated parent map.
        next_token = torch.arange(1, steps + 1, device="xpu").view(1, steps)
        next_token[:, -1] = -1
        siblings = torch.full_like(next_token, -1)
        parents = torch.empty_like(next_token)

        def verify():
            mixed = causal_conv1d_update(
                qkvz[:, :8192].view(1, steps, 8192).transpose(1, 2),
                cs,
                weight,
                activation="silu",
                conv_state_indices=indices,
                intermediate_conv_window=conv_steps,
                intermediate_state_indices=intermediate_indices,
                retrieve_next_token=next_token,
                retrieve_next_sibling=siblings,
                retrieve_parent_token=parents,
                fp32_products=True,
            ).transpose(1, 2)
            q, k, v = mixed.split((2048, 2048, 4096), dim=-1)
            return fused_sigmoid_gating_delta_rule_update(
                A_log=alog,
                a=ba[:, 32:],
                dt_bias=bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q.reshape(1, steps, 16, 128),
                k=k.reshape(1, steps, 16, 128),
                v=v.reshape(1, steps, 32, 128),
                b=ba[:, :32],
                initial_state_source=ss,
                initial_state_indices=indices,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                intermediate_states_buffer=ssm_steps,
                intermediate_state_indices=intermediate_indices,
                cache_steps=steps,
                retrieve_parent_token=parents,
                round_gdn_gate_to_bf16=True,
            )

        actual = verify()
        if captured:
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                actual = verify()
            cs.copy_(cs0)
            ss.copy_(ss0)
            graph.replay()
        torch.xpu.synchronize()
        torch.testing.assert_close(ss, ss0, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.reshape(steps, 32, 128),
            torch.cat(ref_out),
            rtol=0.008,
            atol=1e-4,
        )
        torch.testing.assert_close(conv_steps[1], torch.stack(ref_conv), rtol=0, atol=0)
        torch.testing.assert_close(
            ssm_steps[1], torch.stack(ref_state), rtol=1e-5, atol=2e-6
        )
        caches = SimpleNamespace(
            temporal=ss.unsqueeze(0),
            conv=[cs.unsqueeze(0)],
            intermediate_ssm=ssm_steps.unsqueeze(0),
            intermediate_conv_window=[conv_steps.unsqueeze(0)],
        )
        # commit_len=1 covers all drafted tokens rejected; steps covers all accepted.
        for accepted in range(1, steps + 1):
            scatter_mamba_states_after_mtp_verify(
                caches,
                indices,
                torch.tensor([accepted - 1], device="xpu", dtype=torch.int64),
                mamba_track_indices=None,
                mamba_steps_to_track=None,
                src_indices_raw=intermediate_indices,
            )
            torch.testing.assert_close(cs[2], ref_conv[accepted - 1], rtol=0, atol=0)
            torch.testing.assert_close(
                ss[2], ref_state[accepted - 1], rtol=1e-5, atol=2e-6
            )

    def _reference(self, args):
        q, ba, cs, ss, w, alog, bias, idx = [t.cpu() for t in args]
        slot = int(idx[0])
        # Match the FP32 Conv1D products before its BF16 output boundary.
        history = torch.cat([cs[slot].float(), q[:, :8192].float().t()], 1)
        c = torch.zeros(8192)
        for tap in range(4):
            c += history[:, tap] * w[:, tap].float()
        c = F.silu(c).bfloat16().double()
        qn, kn, v = c.split([2048, 2048, 4096])
        qn, kn = qn.reshape(16, 128), kn.reshape(16, 128)
        qn = qn / torch.sqrt(qn.square().sum(-1, keepdim=True) + 1e-6) / 128**0.5
        kn = kn / torch.sqrt(kn.square().sum(-1, keepdim=True) + 1e-6)
        qn, kn = qn.repeat_interleave(2, 0), kn.repeat_interleave(2, 0)
        decay_logit = (ba[0, 32:].float() + bias.float()).bfloat16().double()
        decay = torch.exp(-torch.exp(alog.double()) * F.softplus(decay_logit))
        beta = torch.sigmoid(ba[0, :32].double())
        h = ss[slot].double() * decay[:, None, None]
        delta = (v.reshape(32, 128) - (h * kn[:, None, :]).sum(-1)) * beta[:, None]
        h += delta[:, :, None] * kn[:, None, :]
        out = (h * qn[:, None, :]).sum(-1).view(1, 32, 128).bfloat16()
        return out, h

    def _check(self, args, reference=False):
        original = [t.clone() for t in args]
        native = [t.clone() for t in args]
        expected, z = self._native(native)
        actual, az = xpu_gdn_decode(*args)
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-6)
        torch.testing.assert_close(az, z, rtol=0, atol=0)
        torch.testing.assert_close(args[2], native[2], rtol=0, atol=0)
        torch.testing.assert_close(args[3], native[3], rtol=3e-5, atol=1e-6)
        if reference:
            ro, rs = self._reference(original)
            torch.testing.assert_close(actual.cpu(), ro, rtol=0.008, atol=1e-6)
            torch.testing.assert_close(
                args[3][int(args[-1][0])].cpu().double(), rs, rtol=3e-5, atol=1e-6
            )

    def test_native_and_independent_reference(self):
        args = self._inputs()
        for mode in ("random", "zero", "nonuniform"):
            if mode == "zero":
                args[0].zero_()
            elif mode == "nonuniform":
                args[0].normal_()
                args[0][:, :128] *= 0.001
            with self.subTest(mode=mode):
                self._check(args, reference=True)

    def test_bf16_decay_logit_sum_boundary(self):
        args = self._inputs()
        args[1][:, 32:].fill_(1.0)
        args[6].fill_(1 / 256)
        self._check(args, reference=True)

    def test_dynamic_graph_state_indices_and_padding(self):
        args = self._inputs()
        native = [t.clone() for t in args]
        for _ in range(2):
            xpu_gdn_decode(*args)
        native[2].copy_(args[2])
        native[3].copy_(args[3])
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            actual, _ = xpu_gdn_decode(*args)
        native[2].copy_(args[2])
        native[3].copy_(args[3])
        for step, slot in enumerate([0, 2, -1, 1, 3, 4, 5, 2]):
            args[0].normal_()
            args[1].normal_()
            args[-1].fill_(slot)
            native[0].copy_(args[0])
            native[1].copy_(args[1])
            native[-1].copy_(args[-1])
            graph.replay()
            if 0 <= slot < 5:
                expected, _ = self._native(native)
            else:
                expected = torch.zeros_like(actual)
            with self.subTest(step=step, slot=slot):
                torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-6)
                torch.testing.assert_close(args[2], native[2], rtol=0, atol=0)
                torch.testing.assert_close(args[3], native[3], rtol=3e-5, atol=1e-6)

    def test_layout_guard(self):
        q, ba, cs, ss, w, alog, bias, _ = self._inputs()
        self.assertTrue(can_use_xpu_gdn_decode(q, ba, cs, ss, w, None, alog, bias))
        self.assertFalse(
            can_use_xpu_gdn_decode(q.repeat(2, 1), ba, cs, ss, w, None, alog, bias)
        )
        self.assertFalse(
            can_use_xpu_gdn_decode(
                q, ba, cs, ss, w, torch.zeros(8192, device="xpu"), alog, bias
            )
        )
        self.assertFalse(
            can_use_xpu_gdn_decode(q, ba, cs, ss, w, None, alog, bias.float())
        )


if __name__ == "__main__":
    unittest.main()
