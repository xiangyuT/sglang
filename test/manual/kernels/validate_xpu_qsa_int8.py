"""Validate block-64 INT8 KV bytes, direct attention, copies and graph replay."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.xpu_qsa_sycl import sparse_attention
from sglang.srt.mem_cache.int8_kv_pool import Int8QSATokenToKVPool


def encode(x):
    # Independent CPU Torch oracle uses IEEE half conversion and ties-to-even.
    f = x.cpu().float().reshape(*x.shape[:2], 4, 64)
    scale = (f.abs().amax(-1) / 127).half()
    divisor = scale.float().unsqueeze(-1)
    quant = (
        torch.where(divisor > 0, (f / divisor).round(), 0)
        .clamp(-127, 127)
        .to(torch.int8)
    )
    return torch.cat((quant.flatten(-2), scale.view(torch.int8)), -1)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    os.environ["SGLANG_XPU_QSA_SYCL_LIB"] = a.library
    os.environ["SGLANG_XPU_GGUF_TP_EP"] = "1"
    torch.manual_seed(192)
    records = []
    for device in ("xpu:0", "xpu:1"):
        torch.xpu.set_device(device)
        for heads, qheads in ((2, 24), (1, 12), (1, 6)):
            pool = Int8QSATokenToKVPool(
                size=192,
                page_size=64,
                dtype=torch.int8,
                head_num=heads,
                head_dim=256,
                layer_num=2,
                device=device,
                enable_memory_saver=False,
            )
            source = torch.randn(66, heads, 256, device=device).bfloat16()
            source[4].zero_()
            source[1].fill_(2**-22)
            source[2].mul_(100)
            source[6].fill_(2**-30)
            source[8].fill_(8e6)
            source[3, :, :4] = torch.tensor(
                [127, 0.5, 1.5, -2.5], device=device
            ).bfloat16()
            key = source[::2]
            value = -source[1::2]
            locations = torch.tensor([*range(32), 0], device=device, dtype=torch.int64)
            layer = SimpleNamespace(layer_id=0)
            pool.set_kv_buffer(layer, locations, key, value)
            rawk, rawv = pool.quant_method.raw_k[0], pool.quant_method.raw_v[0]
            ek, ev = encode(key), encode(value)
            expectedk = torch.zeros_like(rawk, device="cpu")
            expectedv = torch.zeros_like(rawv, device="cpu")
            for i, dst in enumerate(locations.cpu().tolist()):
                expectedk[dst] = ek[i]
                expectedv[dst] = ev[i]
            assert torch.equal(rawk.cpu(), expectedk), "quant K bytes"
            assert torch.equal(rawv.cpu(), expectedv), "quant V bytes"
            assert pool.quant_method.error.item() == 0
            info = pool.get_device_pool_infos()[0].buffer_info
            info.validate()
            assert info.head_dim == 264 and info.scales is None
            assert sum(pool.get_kv_size_bytes()) == 2 * 2 * 256 * heads * 264
            # Cache move includes scales, including overlapping source/destination.
            dst = torch.tensor([4, 5, 6], device=device)
            src = torch.tensor([5, 6, 4], device=device)
            pool.move_kv_cache(dst, src)
            assert torch.equal(rawk[dst].cpu(), expectedk[src.cpu()])
            cp = pool.get_cpu_copy(dst)
            saved = rawk[dst].cpu().clone()
            rawk[dst] = 0
            rawv[dst] = 0
            pool.load_cpu_copy(cp, dst)
            assert torch.equal(rawk[dst].cpu(), saved)
            k, v = pool.get_kv_buffer(0)

            # Compare direct INT8 attention with BF16 native attention on exactly
            # representable dequantized BF16 values; otherwise isolate quant error
            # using a float32 CPU softmax reference of the stored cache.
            def decode(raw):
                return raw[:, :, :256].float().reshape(256, heads, 4, 64) * raw[
                    :, :, 256:
                ].contiguous().view(torch.float16).float().unsqueeze(-1)

            dk = decode(rawk.cpu()).flatten(-2)
            dv = decode(rawv.cpu()).flatten(-2)
            for bs, width in ((1, 70), (2, 70), (4, 129), (8, 2111), (32, 70)):
                q = torch.randn(bs, qheads, 256, device=device).bfloat16() * 0.01
                slots = (
                    torch.arange(width, device=device, dtype=torch.int32) % 32
                ).repeat(bs, 1)
                slots[:, ::7] = -1
                slots[0, :] = -1
                out = sparse_attention(q, k, v, slots, int8=True)
                ref = torch.zeros_like(q, device="cpu")
                qq = q.cpu().float()
                for row in range(1, bs):
                    ids = slots[row].cpu().long()
                    ids = ids[ids >= 0]
                    keys = (
                        dk[ids]
                        .repeat_interleave(qheads // heads, dim=1)
                        .permute(1, 0, 2)
                    )
                    values = (
                        dv[ids]
                        .repeat_interleave(qheads // heads, dim=1)
                        .permute(1, 0, 2)
                    )
                    prob = torch.softmax(
                        (qq[row, :, None, :] * keys).sum(-1) / 16, dim=-1
                    )
                    ref[row] = (prob[:, :, None] * values).sum(1).bfloat16()
                torch.testing.assert_close(
                    out.cpu().float(), ref.float(), atol=0.03125, rtol=0.015625
                )
            # Replay must use new source values and location contents, not capture values.
            kin = torch.randn(8, heads, 256, device=device).bfloat16()
            vin = -kin.clone()
            loc = torch.arange(8, device=device, dtype=torch.int32) + 64
            pool.set_kv_buffer(layer, loc, kin, vin)
            torch.xpu.synchronize()
            print(f"capture {device} H{heads} Q{qheads}", flush=True)
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph, stream=torch.xpu.Stream(device=device)):
                pool.set_kv_buffer(layer, loc, kin, vin)
                y = sparse_attention(q[:8], k, v, slots[:8], int8=True)
            kin.mul_(0.25)
            vin.mul_(2)
            loc.add_(8)
            graph.replay()
            torch.xpu.synchronize()
            assert torch.equal(rawk[loc.long()].cpu(), encode(kin))
            torch.testing.assert_close(
                y, sparse_attention(q[:8], k, v, slots[:8], int8=True), atol=0, rtol=0
            )
            # Large prefill ownership pass after graph-sized writes, including
            # invalid locations and duplicates that cross the 8192-row boundary.
            big = torch.randn(8201, heads, 256, device=device).bfloat16()
            loc_big = (torch.arange(8201, device=device, dtype=torch.int64) % 240) + 8
            loc_big[0] = -1
            loc_big[8191] = 99999
            expected = rawk.cpu().clone()
            pool.set_kv_buffer(layer, loc_big, big, -big)
            encoded = encode(big)
            for row, target in enumerate(loc_big.cpu().tolist()):
                if 0 <= target < 256:
                    expected[target] = encoded[row]
            assert torch.equal(rawk.cpu(), expected)
            # Nonfinite inputs trigger the sticky flag and poisoned scale, no UB.
            kin[0, 0, 0] = float("nan")
            pool.set_kv_buffer(layer, loc, kin, vin)
            assert pool.quant_method.error.item() == 1
            records.append(
                dict(
                    device=device,
                    uuid=str(torch.xpu.get_device_properties(device).uuid),
                    heads=heads,
                    qheads=qheads,
                    bytes_exact=True,
                    attention=True,
                    graph=True,
                    copies=True,
                )
            )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records))


if __name__ == "__main__":
    run()
