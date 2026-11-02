"""Native GDN radix checkpoints must survive independent branch updates."""
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")


@pytest.mark.parametrize("lengths,cuts,encoded_lengths", [
    ([256], [256], [256]),
    ([320], [256], [256]),
    ([320, 256], [256, 256], [256, 256]),
    ([64]*8, [64]*8, [64]*8),
    # The scheduler passes an unaligned end to select an intermediate state,
    # or boundary+1 when forcing an interior aligned checkpoint. Neither is
    # the actual state depth stored in the radix tree.
    ([318], [256], [318]),
    ([320], [256], [257]),
    ([318, 320, 512, 64], [256, 256, 512, 64], [318, 257, 512, 64]),
])
@torch.inference_mode()
def test_native_prefill_checkpoint_and_branch_isolation(lengths, cuts, encoded_lengths, monkeypatch):
    from sglang.srt.hardware_backend.xpu.attention.xpu_gdn_backend import XpuGDNAttnBackend
    monkeypatch.setattr(
        "sglang.srt.hardware_backend.xpu.attention.xpu_gdn_backend.mamba_cache_chunk_size",
        lambda: 64,
        raising=False,
    )

    torch.manual_seed(719)
    b = len(lengths)
    heads, keys, width = 24, 8, 5120
    qkvz = torch.randn(sum(lengths), 8192, device="xpu", dtype=torch.bfloat16) * .1
    ba = torch.randn(sum(lengths), 48, device="xpu", dtype=torch.bfloat16) * .1
    conv = torch.randn(2*b+1, width, 3, device="xpu", dtype=torch.bfloat16) * .1
    ssm = torch.randn(2*b+1, heads, 128, 128, device="xpu") * .01
    layer = NS(num_k_heads=keys, num_v_heads=heads, head_k_dim=128, head_v_dim=128,
               conv_weights=torch.randn(width, 4, device="xpu", dtype=torch.bfloat16)*.03,
               bias=None, activation="silu", A_log=torch.rand(heads, device="xpu")*4-2,
               dt_bias=torch.rand(heads, device="xpu", dtype=torch.bfloat16))
    ids = torch.arange(1, b+1, device="xpu", dtype=torch.int32)
    tracks = torch.arange(b+1, 2*b+1, device="xpu", dtype=torch.int32)
    qsl = torch.tensor([0]+list(torch.tensor(lengths).cumsum(0)), device="xpu", dtype=torch.int32)
    obj = object.__new__(XpuGDNAttnBackend)
    obj.forward_metadata = NS(query_start_loc=qsl)
    fb = NS(batch_size=b, mamba_prefill_track_mask_cpu=[True]*b,
            extend_seq_lens_cpu=lengths, extend_prefix_lens_cpu=[64]*b,
            mamba_track_seqlens_cpu=[64+c for c in encoded_lengths], mamba_track_indices=tracks,
            extend_prefix_lens=torch.full((b,),64,device="xpu"))
    cref, sref = conv.clone(), ssm.clone()
    output, z = obj._forward_tracked_prefill(layer, fb, qkvz, ba, conv, ssm, ids)
    offset = 0
    for row,(length,cut) in enumerate(zip(lengths,cuts)):
        end = offset+cut
        ref,_ = obj._run_pool_gdn(layer,qkvz[offset:end],ba[offset:end],cref,sref,ids[row:row+1],
            torch.tensor([0,cut],device="xpu",dtype=torch.int32),torch.ones(1,device="xpu",dtype=torch.bool),
            num_prefills=1,num_decodes=0)
        assert torch.equal(ssm[b+1+row].view(torch.int32), sref[1+row].view(torch.int32))
        assert torch.equal(conv[b+1+row].view(torch.int16), cref[1+row].view(torch.int16))
        assert torch.equal(output[offset:end].view(torch.int16),ref.view(torch.int16))
        if cut<length:
            obj._run_pool_gdn(layer,qkvz[end:offset+length],ba[end:offset+length],cref,sref,ids[row:row+1],
                torch.tensor([0,length-cut],device="xpu",dtype=torch.int32),torch.ones(1,device="xpu",dtype=torch.bool),
                num_prefills=1,num_decodes=0)
        assert torch.equal(ssm[1+row].view(torch.int32), sref[1+row].view(torch.int32))
        assert torch.equal(conv[1+row].view(torch.int16), cref[1+row].view(torch.int16))
        offset+=length
