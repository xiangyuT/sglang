"""Host round trips preserve hybrid checkpoints, including PLE and draft rows."""
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")


@pytest.mark.parametrize("packed_draft", (False, True))
@pytest.mark.parametrize("count", (1,8))
@torch.inference_mode()
def test_host_roundtrip_with_shuffled_slots_and_side_states(monkeypatch, packed_draft, count):
    from sglang.srt import runtime_context
    from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
    from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
    monkeypatch.setattr(runtime_context,"get_memory",lambda:NS(hicache_io_backend="direct"))
    torch.manual_seed(41)
    slots = 12
    def make(layers):
        return MambaStateBufferInfo(
            temporal=torch.randn(layers,slots,2,16,16,device="xpu"),
            conv=(torch.randn(layers,slots,64,3,device="xpu",dtype=torch.bfloat16),),
            extra_checkpoint_buffers=(torch.randint(100000,(slots,6),device="xpu"),
                                      torch.randn(slots,32,3,device="xpu",dtype=torch.bfloat16)))
    info=make(3);drafts=(make(1),)if packed_draft else ()
    host=MambaPoolHost(info,2,0,device_capacity=slots,layout="page_first_direct",mtp_draft_buffer_infos=drafts)
    source=torch.arange(count-1,-1,-1,device="xpu",dtype=torch.int64)
    hid=host.alloc(count).flip(0)
    dest=torch.arange(2,2+count,device="xpu",dtype=torch.int64)
    expected=[]
    for item in (info,*drafts):
        for state in (item.temporal,*item.conv):expected.append(state[:,source].cpu().clone())
        for state in item.extra_checkpoint_buffers:expected.append(state[source].cpu().clone())
    host.backup_from_device_all_layer(None,hid,source,"direct")
    torch.xpu.synchronize()
    for item in (info,*drafts):
        for state in (item.temporal,*item.conv,*item.extra_checkpoint_buffers):state.zero_()
    for layer in range(host.num_mamba_layers):host.load_to_device_per_layer(None,hid,dest,layer,"direct")
    torch.xpu.synchronize()
    actual=[]
    for item in (info,*drafts):
        for state in (item.temporal,*item.conv):actual.append(state[:,dest].cpu())
        for state in item.extra_checkpoint_buffers:actual.append(state[dest].cpu())
    assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8))for a,b in zip(actual,expected))
    host.free(hid);host.destroy()
