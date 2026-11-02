# SPDX-License-Identifier: Apache-2.0
import struct
import pytest
import torch
from sglang.srt.layers.quantization.gguf_expert_profile import read_profile,initial_placement,read_experts,growth_placement


def profile_file(tmp_path,pairs):
    f=tmp_path/'profile.bin';f.write_bytes(b'STRP'+struct.pack('<5I',1,2,4,8,len(pairs))+b''.join(struct.pack('<HH',*x)for x in pairs));return f


def test_incomplete_ranking_fills_each_layer_once(tmp_path):
    profile=read_profile(str(profile_file(tmp_path,[(0,3),(1,2),(0,1)])))
    assert profile['ranked']==[[3,1,0,2],[2,0,1,3]]
    resident,mirror,lru=initial_placement(profile,0,0,4,2)
    assert resident==(3,1)and mirror==(0,2)and lru==(1,3)


def test_partition_uses_local_ids_and_keeps_dummy_resident(tmp_path):
    profile=read_profile(str(profile_file(tmp_path,[(0,3),(0,2)])))
    resident,mirror,lru=initial_placement(profile,0,2,2,2,dummy=2)
    assert resident==(1,2)and mirror==(0,)and lru==(1,2)
    assert set(resident)|set(mirror)==set(range(3))


@pytest.mark.parametrize('pairs',[[(0,1),(0,1)],[(2,0)],[(0,4)]])
def test_invalid_rankings_fail_before_weights_are_reordered(tmp_path,pairs):
    with pytest.raises(ValueError):read_profile(str(profile_file(tmp_path,pairs)))


def test_truncated_profile_rejected(tmp_path):
    f=profile_file(tmp_path,[(0,1)]);f.write_bytes(f.read_bytes()[:-1])
    with pytest.raises(ValueError):read_profile(str(f))


def test_noncontiguous_expert_order_preserves_encoded_rows():
    data=torch.arange(32,dtype=torch.uint8).view(8,4)
    class Source:
        row_bytes=4
        def packed_row_range(self,first,last):return data[first:last].clone()
    out=read_experts(Source(),[2,0,3],2,pin_memory=False)
    assert torch.equal(out,torch.stack([data[4:6].flatten(),data[:2].flatten(),data[6:8].flatten()]))


def test_growth_preserves_current_owners_and_profile_rank(tmp_path):
    profile=read_profile(str(profile_file(tmp_path,[(0,3),(0,1),(0,2),(0,0)])))
    assert growth_placement(profile,0,0,4,[0,1,-1,-1],2)==(3,2)
    assert growth_placement(profile,0,0,4,[0,1,-1,-1],0)==()


def test_growth_partition_uses_local_ids(tmp_path):
    profile=read_profile(str(profile_file(tmp_path,[(0,3),(0,2)])))
    assert growth_placement(profile,0,2,2,[0,-1,1],1,dummy=2)==(1,)


@pytest.mark.parametrize('current,count',[([0,-1],1),([0,1,-1,-1],3),([0,1,-1,-1],-1)])
def test_invalid_growth_rejected(tmp_path,current,count):
    profile=read_profile(str(profile_file(tmp_path,[])))
    with pytest.raises(ValueError):growth_placement(profile,0,0,4,current,count)
