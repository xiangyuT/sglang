"""Run under torchrun --nproc_per_node=2 on the selected B70 TP2 pair."""
import datetime
import os

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    not os.environ.get('SGLANG_XPU_GGUF_TP_GREEDY_LIB') or
    not os.environ.get('SGLANG_XPU_PEER_ALLREDUCE_LIB') or
    os.environ.get('WORLD_SIZE') != '2',
    reason='Native B70 greedy library and a two-rank peer test required',
)


@pytest.fixture(scope='module')
def peer():
    from sglang.srt.distributed.device_communicators.xpu_peer_allreduce import PeerAllReduce
    rank = int(os.environ['LOCAL_RANK'])
    torch.xpu.set_device(rank)
    torch.set_num_threads(1)
    assert torch.xpu.get_device_properties().device_id == 0xE223
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=120))
    value = PeerAllReduce(dist.group.WORLD, os.environ['SGLANG_XPU_PEER_ALLREDUCE_LIB'])
    yield value
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize('rows', [1, 4, 8, 12, 16, 20, 24, 28, 32])
@pytest.mark.parametrize('kind', ['random', 'ties', 'nonfinite'])
@torch.inference_mode()
def test_rows_ties_nans_live_values_and_graph(peer, rows, kind):
    from sglang.srt.layers.quantization.gguf_tp_greedy import compact_argmax, _native_exchange
    width = 124160
    gen = torch.Generator().manual_seed(792 + rows)
    data = torch.randn((2, rows, width), generator=gen)
    if kind == 'ties':
        data.fill_(-1)
        data[:, :, 0] = 0
        data[0, :, 0] = -0.0
        data[:, :, -1] = 0
        if rows > 1:
            data[1, 1, 17] = 2
    elif kind == 'nonfinite':
        for row in range(rows):
            mode = row % 4
            if mode == 0:
                data[:, row].fill_(-float('inf'))
            elif mode == 1:
                data[0, row, -1] = float('inf')
                data[1, row, 0] = float('inf')
            elif mode == 2:
                data[0, row, -1] = float('nan')
                data[1, row, 0] = float('nan')
            else:
                data[1, row, 17] = float('nan')
    x = data[peer.rank].to('xpu')
    compact_argmax(peer, x)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        indices = compact_argmax(peer, x)
        carrier = _native_exchange(peer, x, carrier=True)
    for value in (data, data.flip(-1), data.roll(31, dims=-1)):
        x.copy_(value[peer.rank])
        graph.replay()
        want = torch.cat(list(value.unbind(0)), dim=1).argmax(dim=1)
        assert torch.equal(indices.cpu(), want)
        expected = torch.full((rows, width*2), -float('inf'))
        expected.scatter_(1, want[:,None], 1.)
        assert torch.equal(carrier.cpu(), expected)
