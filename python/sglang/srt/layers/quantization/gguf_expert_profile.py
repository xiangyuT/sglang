# SPDX-License-Identifier: Apache-2.0
"""Read the fixed upstream Strata ranking for initial compressed residency."""
from functools import lru_cache
from pathlib import Path
import hashlib
import struct

@lru_cache(None)
def read_profile(path):
    data=Path(path).read_bytes()
    if len(data)<24 or data[:4]!=b'STRP':raise ValueError('Invalid STRP expert profile')
    version,layers,experts,slots,count=struct.unpack_from('<5I',data,4)
    if version!=1 or layers<1 or experts<1 or count>slots or count>layers*experts or len(data)<24+4*count:
        raise ValueError('Unsupported or truncated STRP expert profile')
    pairs=struct.unpack_from('<'+'H'*(2*count),data,24)
    ranks=[[]for _ in range(layers)];seen=set()
    for layer,expert in zip(pairs[::2],pairs[1::2]):
        if layer>=layers or expert>=experts or (layer,expert)in seen:raise ValueError('Invalid or duplicate expert profile pair')
        seen.add((layer,expert));ranks[layer].append(expert)
    for layer,row in enumerate(ranks):row.extend(e for e in range(experts)if(layer,e)not in seen)
    return dict(layers=layers,experts=experts,sha256=hashlib.sha256(data).hexdigest(),ranked=ranks)


def initial_placement(profile,layer,start,real,slots,*,dummy=None):
    if not (0<=layer<profile['layers'] and 0<=start<profile['experts'] and 1<=real<=profile['experts']-start
            and 1<=slots<=real+(dummy is not None) and (dummy is None or dummy==real)):
        raise ValueError('Invalid expert profile partition or slot capacity')
    rank=[e-start for e in profile['ranked'][layer]if start<=e<start+real]
    resident=rank[:slots-(dummy is not None)]
    if dummy is not None:resident.append(dummy)
    resident_set=set(resident)
    mirror=[e for e in range(real+(dummy is not None))if e not in resident_set]
    lru=list(reversed([e for e in resident if e!=dummy]))+([]if dummy is None else[dummy])
    return tuple(resident),tuple(mirror),tuple(lru)


def read_experts(source,experts,rows_per_expert,*,pin_memory=True):
    import torch
    output=torch.empty((len(experts),rows_per_expert*source.row_bytes),dtype=torch.uint8,pin_memory=pin_memory)
    for index,expert in enumerate(experts):
        output[index].copy_(source.packed_row_range(expert*rows_per_expert,(expert+1)*rows_per_expert).view(-1))
    return output


def growth_placement(profile,layer,start,real,current,count,*,dummy=None):
    """Promote currently mirrored experts in the original profile order.

    Graph warmups may have exchanged owners since initial placement. Select
    from the live map and retain every existing resident slot.
    """
    if (len(current)!=real+(dummy is not None) or count<0
            or count>sum(slot<0 for slot in current)):
        raise ValueError('Invalid profile-driven cache growth')
    ranked,_,_=initial_placement(profile,layer,start,real,len(current),dummy=dummy)
    return tuple(e for e in ranked if current[e]<0)[:count]
