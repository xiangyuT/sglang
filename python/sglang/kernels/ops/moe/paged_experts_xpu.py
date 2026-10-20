"""XPU-resident expert planning and copies from pinned USM host stores.

No tensor values are read by Python. Both kernels accept persistent device
buffers and are safe to capture with torch.xpu.XPUGraph.
"""

import triton
import triton.language as tl


@triton.jit
def _decide_kernel(
    Topk,
    Weights,
    Step,
    SlotExpert,
    ExpertSlot,
    LastUse,
    Src,
    Dst,
    Count,
    MappedIds,
    MappedWeights,
    E: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    s = tl.arange(0, BK)
    t = tl.arange(0, BT)
    experts = tl.load(SlotExpert + s, s < K, other=-2)
    age = tl.load(LastUse + s, s < K, other=2147483647)
    old_experts = experts
    routed = tl.load(Topk + t, t < T, other=-1).to(tl.int32)
    valid = (t < T) & (routed >= 0) & (routed < E)
    now = tl.load(Step) + 1
    needed = (
        tl.sum(((experts[:, None] == routed[None, :]) & valid[None, :]).to(tl.int32), 1)
        > 0
    )
    age = tl.where(needed, now, age)
    n = 0
    for i in range(T):
        expert = tl.load(Topk + i).to(tl.int32)
        present = tl.sum(((experts == expert) & (s < K)).to(tl.int32), 0) > 0
        if (expert >= 0) & (expert < E) & ~present:
            eligible = (s < K) & (age != now)
            oldest = tl.min(tl.where(eligible, age, 2147483647), 0)
            victim = tl.min(tl.where(eligible & (age == oldest), s, K), 0)
            tl.store(Src + n, expert)
            tl.store(Dst + n, victim)
            n += 1
            experts = tl.where(s == victim, expert, experts)
            age = tl.where(s == victim, now, age)
    # Clear the old bijection before writing its replacement. One program owns
    # the whole layer's state; the barrier orders global stores across warps.
    tl.store(ExpertSlot + old_experts, -1, s < K)
    tl.debug_barrier()
    tl.store(ExpertSlot + experts, s, s < K)
    tl.store(SlotExpert + s, experts, s < K)
    tl.store(LastUse + s, age, s < K)
    tl.store(Step, now)
    tl.store(Count, n)
    match = (experts[:, None] == routed[None, :]) & (s[:, None] < K)
    mapped = tl.max(tl.where(match, s[:, None], -1), 0)
    weights = tl.load(Weights + t, t < T, other=0)
    tl.store(MappedIds + t, tl.where(valid, mapped, 0), t < T)
    tl.store(MappedWeights + t, tl.where(valid, weights, 0), t < T)


@triton.jit
def _gather_kernel(
    Hosts,
    Gpus,
    Src,
    Dst,
    Count,
    ROWS: tl.constexpr,
    GRID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    count = tl.load(Count)
    if count > 0:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        stride: tl.constexpr = GRID * BLOCK
        for p in tl.static_range(len(ROWS)):
            row = ROWS[p]
            host = tl.load(Hosts + p).to(tl.pointer_type(tl.int32))
            gpu = tl.load(Gpus + p).to(tl.pointer_type(tl.int32))
            total = count * row
            for j in range(tl.cdiv(total, stride)):
                position = offsets + j * stride
                plan = position // row
                within = position % row
                active = position < total
                src = tl.load(Src + plan, active, other=0)
                dst = tl.load(Dst + plan, active, other=0)
                value = tl.load(host + src * row + within, active, other=0)
                tl.store(gpu + dst * row + within, value, active)


def decide(topk, weights, residency, mapped_ids, mapped_weights):
    entries = topk.numel()
    _decide_kernel[(1,)](
        topk,
        weights,
        residency.step,
        residency.slot_expert,
        residency.expert_slot,
        residency.slot_lastuse,
        residency.src,
        residency.dst,
        residency.count,
        mapped_ids,
        mapped_weights,
        E=residency.expert_slot.numel(),
        K=residency.slot_expert.numel(),
        T=entries,
        BK=triton.next_power_of_2(residency.slot_expert.numel()),
        BT=triton.next_power_of_2(entries),
        num_warps=4,
    )


def gather(hosts, gpus, rows, residency):
    _gather_kernel[(128,)](
        hosts,
        gpus,
        residency.src,
        residency.dst,
        residency.count,
        ROWS=rows,
        GRID=128,
        BLOCK=1024,
        num_warps=4,
    )
