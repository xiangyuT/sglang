"""Native decode metadata for a single causal DFlash verification chain."""

import torch


def supports_dflash_verify_layout(
    q, key_cache, value_cache, page_table, cache_seqlens, max_seqlen_q
):
    """Shape-only contract shared by the experimental native verify paths."""
    return (
        q.device.type == "xpu"
        and q.dtype == key_cache.dtype == value_cache.dtype == torch.bfloat16
        and q.ndim == 3
        and 2 <= q.shape[0] <= 8
        and q.shape[1:] == (16, 256)
        and max_seqlen_q == q.shape[0]
        and key_cache.ndim == value_cache.ndim == 4
        and key_cache.shape == value_cache.shape
        and key_cache.shape[1:] == (128, 2, 256)
        and page_table.ndim == 2
        and page_table.shape[0] == 1
        and page_table.dtype == cache_seqlens.dtype == torch.int32
        and cache_seqlens.shape == (1,)
        and all(
            t.device == q.device and t.is_contiguous()
            for t in (q, key_cache, value_cache, page_table, cache_seqlens)
        )
    )


def make_dflash_verify_decode_metadata(
    q, key_cache, value_cache, page_table, cache_seqlens, max_seqlen_q
):
    """Reframe one causal chain; query i sees prefix+i+1 slots.

    The caller guarantees that the current K/V are already written. Separate
    one-query batches preserve the causal mask but change native reduction
    rounding. All lengths stay on the device, including during graph capture.
    """
    if not supports_dflash_verify_layout(
        q, key_cache, value_cache, page_table, cache_seqlens, max_seqlen_q
    ):
        return None
    m = q.shape[0]
    positions = torch.arange(m + 1, device=q.device, dtype=torch.int32)
    lengths = cache_seqlens - m + positions[1:]
    # The native kernel requires contiguous page tables, including batch stride.
    return page_table.repeat(m, 1), lengths, positions
