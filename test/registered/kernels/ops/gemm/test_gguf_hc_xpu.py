"""Preserve original HC parameters and full outputs across eager/graph phases."""

import os
import unittest

import pytest
import torch


@pytest.mark.parametrize('tokens', [5, 8, 1024, 4096])
def test_prefill_hc_norm_preserves_grouped_gemma_bits(monkeypatch, tokens):
    if not torch.xpu.is_available() or not os.getenv('SGLANG_XPU_GGUF_HC_PREFILL_NORM_LIB'):
        pytest.skip('requires the B70 prefill HC normalization library')
    from sglang.srt.layers.hyperconnection import GroupedGemmaRMSNorm

    torch.manual_seed(71 + tokens)
    x = torch.randn(tokens, 10240, device='xpu', dtype=torch.bfloat16)
    module = GroupedGemmaRMSNorm(10240, eps=1e-6, group_size=2560).to(
        device='xpu', dtype=torch.bfloat16
    )
    with torch.no_grad():
        module.weight.copy_(torch.randn_like(module.weight) * .2)
        reference = module(x)
        module._xpu_gguf_hc_enabled = True
        monkeypatch.setenv('SGLANG_XPU_GGUF_HC_PREFILL_NORM', '1')
        output = module(x)
    assert torch.equal(reference.view(torch.int16), output.view(torch.int16))
from sglang.kernels.ops.gemm.gguf_hc_xpu import install
from sglang.kernels.ops.quantization.gguf_sycl import _library
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=20, suite="xpu-unit-test")


@unittest.skipUnless(torch.xpu.is_available(), "XPU required")
class TestGGUFHC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.xpu.get_device_properties().device_id != 0xE223:
            raise unittest.SkipTest("B70 native HC")
        path = os.environ.get("SGLANG_XPU_GGUF_SYCL_LIB")
        if not path or not hasattr(_library(path), "xpu_gguf_hc_norm"):
            raise unittest.SkipTest("Build/select the HC GGUF library")
        torch.set_num_threads(1)
        torch.manual_seed(42)
        torch.set_grad_enabled(False)
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = False

    def module(self, *, combine=True):
        config = HyperConnectionConfig(
            hc_count=4,
            hidden_size=2560,
            params_dtype=torch.bfloat16,
            hc_lowrank=320,
            hc_per_branch_norm=True,
        )
        # Match the loader's construction scope, including norm Parameters.
        with set_default_torch_dtype(torch.bfloat16), torch.device("xpu"):
            module = GatedResidual(config, use_combine=combine).requires_grad_(False)
        self.assertEqual(module.hc_norm.weight.dtype, torch.bfloat16)
        module.hc_norm.weight.copy_(torch.randn_like(module.hc_norm.weight) * 0.01)
        return module

    def inputs(self, tokens, seed):
        generator = torch.Generator().manual_seed(seed)
        x = (
            torch.randn(tokens, 10240, generator=generator, dtype=torch.float32)
            .bfloat16()
            .to("xpu")
        )
        block = (
            torch.randn(tokens, 2560, generator=generator, dtype=torch.float32)
            .bfloat16()
            .to("xpu")
        )
        return x, block

    def forward(self, module, x, block):
        mixed, state = module.mix(x)
        return mixed, module.combine(block, state)

    def exact(self, actual, expected):
        for a, b in zip(actual, expected):
            self.assertTrue(
                torch.equal(a.cpu().view(torch.int16), b.cpu().view(torch.int16))
            )

    def test_parameters_and_complete_outputs_including_prefill(self):
        module = self.module()
        cases = [self.inputs(t, 42 + t) for t in (1, 2, 3, 4, 7)]
        expected = [
            tuple(v.cpu() for v in self.forward(module, *inputs)) for inputs in cases
        ]
        before = {n: p.detach().cpu().clone() for n, p in module.named_parameters()}
        records = install(module)
        self.assertTrue(records[0]["paired"])
        packed = module._xpu_gguf_hc_down_gate
        self.assertEqual(packed.shape, (324, 10240))
        self.assertEqual(
            module.input_mix_weight_down.weight.data_ptr(), packed.data_ptr()
        )
        self.assertEqual(
            module.block_inject_weight.weight.data_ptr(), packed[320].data_ptr()
        )
        for name, value in module.named_parameters():
            self.assertTrue(
                torch.equal(
                    value.cpu().view(torch.int16), before[name].view(torch.int16)
                )
            )
        for inputs, reference in zip(cases, expected):
            with self.subTest(tokens=inputs[0].shape[0]):
                self.exact(self.forward(module, *inputs), reference)
        install(module)
        self.assertEqual(module._xpu_gguf_hc_down_gate.data_ptr(), packed.data_ptr())

    def test_graph_replay_and_next_normalization_contract(self):
        module = self.module()
        inputs = [self.inputs(4, seed) for seed in (3, 7, 11)]
        references = [
            tuple(v.cpu() for v in self.forward(module, *values)) for values in inputs
        ]
        install(module)
        x, block = (v.clone() for v in inputs[0])
        self.forward(module, x, block)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            mixed, state = module.mix(x)
            out, next_norm = module.combine_and_normalize(block, state, module.hc_norm)
        self.assertIsNone(next_norm)
        for values, expected in zip(inputs, references):
            x.copy_(values[0])
            block.copy_(values[1])
            graph.replay()
            self.exact((mixed, out), expected)

    def test_output_mixer_without_injection(self):
        module = self.module(combine=False)
        x, _ = self.inputs(4, 123)
        expected = module.mix(x)[0].cpu()
        records = install(module)
        self.assertFalse(records[0]["paired"])
        self.exact((module.mix(x)[0],), (expected,))


if __name__ == "__main__":
    unittest.main()
