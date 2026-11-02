# SPDX-License-Identifier: Apache-2.0
"""Opt-in FP32 router logits without expanding BF16 router weights."""

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod


class GGUFRouterFP32Method(UnquantizedLinearMethod):
    def apply(self, layer, x, bias=None):
        # Keep the BF16 inputs and weights; avoid rounding logits before TopK.
        # This covers prefill, decode and target verification alike.
        if (x.device.type == "xpu" and x.dtype == torch.bfloat16
                and layer.weight.dtype == torch.bfloat16):
            output = torch.mm(x, layer.weight.t(), out_dtype=torch.float32)
            return output if bias is None else output + bias.float()
        return F.linear(x.float(), layer.weight.float(),
                        None if bias is None else bias.float())


def install_router_fp32(model):
    """Install only on the target GGUF model's routed-expert gates."""
    records = []
    for layer in model.model.layers:
        gate = layer.mlp.gate
        if type(gate.quant_method) is not UnquantizedLinearMethod:
            raise ValueError("FP32 GGUF router requires an unquantized gate")
        if gate.weight.dtype != torch.bfloat16:
            raise ValueError("FP32 GGUF router currently requires BF16 weights")
        gate.quant_method = GGUFRouterFP32Method()
        records.append(dict(layer=layer.layer_id, weight_dtype=str(gate.weight.dtype),
                            output_dtype=str(torch.float32)))
    return records
