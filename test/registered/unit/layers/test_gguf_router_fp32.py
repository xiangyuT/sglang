"""Router precision must survive the gate while weights stay in BF16."""
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.gguf_router_fp32 import (
    GGUFRouterFP32Method, install_router_fp32,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_router_keeps_logits_that_bf16_output_would_tie():
    x = torch.tensor([[1., 1.]], dtype=torch.bfloat16)
    weight = torch.nn.Parameter(torch.tensor([[1., 0.], [1., 0.001953125]],
                                             dtype=torch.bfloat16), requires_grad=False)
    gate = SimpleNamespace(weight=weight)
    pointer = weight.data_ptr()
    old = torch.nn.functional.linear(x, weight)
    actual = GGUFRouterFP32Method().apply(gate, x)
    assert old[0, 0] == old[0, 1]
    assert actual.dtype == torch.float32
    assert actual.tolist() == [[1., 1.001953125]]
    assert actual.argmax(-1).item() == 1
    assert weight.dtype == torch.bfloat16 and weight.data_ptr() == pointer


def test_install_targets_routed_gates_and_preserves_parameters():
    gate = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(3, 2, dtype=torch.bfloat16)),
                           quant_method=UnquantizedLinearMethod())
    shared_gate = object()
    layer = SimpleNamespace(layer_id=7, mlp=SimpleNamespace(gate=gate, shared_expert_gate=shared_gate))
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    weight = gate.weight
    receipt = install_router_fp32(model)
    assert type(gate.quant_method) is GGUFRouterFP32Method
    assert gate.weight is weight and layer.mlp.shared_expert_gate is shared_gate
    assert receipt == [dict(layer=7, weight_dtype='torch.bfloat16', output_dtype='torch.float32')]
    actual = gate.quant_method.apply(gate, torch.ones(2, 2, dtype=torch.bfloat16),
                                     torch.tensor([.25, .5, .75]))
    torch.testing.assert_close(actual, torch.tensor([[2.25, 2.5, 2.75]]).expand(2, -1))
