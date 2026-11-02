"""Opt-in B1 pure-AR paged128 decode preparation; immutable per-forward outputs."""
import ctypes
import os
import torch

ENABLED = os.environ.get('CODEX_AR231_PREPARE') == '1'
LIB = None
ERRORS = None
CALLS = 0
PAGES = 0

def _eligible(batch, token_per_req):
    allocator = batch.tree_cache.token_to_kv_pool_allocator
    pool = batch.req_to_token_pool
    tensors = [batch.req_pool_indices,batch.seq_lens,batch.orig_seq_lens]
    return (
        ENABLED and token_per_req == 1 and batch.forward_mode.is_decode()
        and batch.spec_algorithm.is_none() and batch.spec_info is None
        and len(batch.reqs) == 1 and not batch.model_config.is_encoder_decoder
        and batch.hisparse_coordinator is None
        and type(allocator).__name__ == 'PagedTokenToKVPoolAllocator'
        and type(pool).__name__ == 'HybridReqToTokenPool'
        and allocator.page_size == batch.tree_cache.page_size == 128
        and not allocator.debug_mode
        and all(t is not None and t.shape == (1,) and t.is_contiguous() and t.device.type == 'xpu' for t in tensors)
        and all(t.device == batch.seq_lens.device for t in tensors)
        and tensors[0].dtype == tensors[1].dtype == torch.int64 and tensors[2].dtype == torch.int32
        and pool.req_to_token.ndim == 2 and pool.req_to_token.is_contiguous()
        and pool.req_to_token.dtype == torch.int32 and pool.req_to_token.device == batch.seq_lens.device
        and allocator.free_pages.dtype == torch.int64 and allocator.free_pages.is_contiguous()
        and allocator.free_pages.device == batch.seq_lens.device
        and batch.seq_lens_cpu is not None and batch.seq_lens_cpu.shape == (1,)
        and batch.req_pool_indices_cpu is not None and batch.req_pool_indices_cpu.shape == (1,)
    )

def try_prepare(batch, token_per_req):
    global LIB, ERRORS, CALLS, PAGES
    if not _eligible(batch,token_per_req):
        return None
    pool = batch.req_to_token_pool
    allocator = batch.tree_cache.token_to_kv_pool_allocator
    old_length = int(batch.seq_lens_cpu[0])
    row = int(batch.req_pool_indices_cpu[0])
    if not (0 <= row < pool.req_to_token.shape[0] and 1 <= old_length < pool.req_to_token.shape[1]):
        return None
    from sglang.srt.mem_cache.common import evict_from_tree_cache, available_and_evictable_str
    # Keep the original overestimate, staged-page merge and OOM behavior.
    evict_from_tree_cache(batch.tree_cache,128)
    if len(allocator.free_pages) < 1:
        allocator.merge_and_sort_free()
    needed = int(old_length % 128 == 0)
    if needed > len(allocator.free_pages):
        message = 'Decode out of memory. Try to lower your batch size.\nTry to allocate 1 tokens.\n' + available_and_evictable_str(batch.tree_cache)
        batch.tree_cache.pretty_print()
        raise RuntimeError(message)
    if LIB is None:
        LIB = ctypes.CDLL('/evidence/tpot-iterations-20261023/native/prepare231/prepare.so')
        LIB.prepare231_run.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*3
        LIB.prepare231_error.restype = ctypes.c_char_p
        ERRORS = torch.zeros(1,dtype=torch.int32,device=batch.seq_lens.device)
    # One fresh allocation per forward: old queued tensors are never overwritten.
    words = torch.empty(5,dtype=torch.int64,device=batch.seq_lens.device)
    seq = words[0:1]
    orig = words[1:2].view(torch.int32)[:1]
    positions = words[2:3]
    out_loc = words[3:4]
    stream = torch.xpu.current_stream(batch.seq_lens.device)
    tensors = [batch.req_pool_indices,batch.seq_lens,batch.orig_seq_lens,
               pool.req_to_token,allocator.free_pages,words,ERRORS]
    rc = LIB.prepare231_run(stream.sycl_queue,*[t.data_ptr() for t in tensors],
        pool.req_to_token.shape[0],pool.req_to_token.shape[1],len(allocator.free_pages))
    if rc:
        raise RuntimeError(LIB.prepare231_error().decode())
    for tensor in tensors:
        tensor.record_stream(stream)
    allocator.free_pages = allocator.free_pages[needed:]
    CALLS += 1
    PAGES += needed
    state = (seq,orig,positions,out_loc,words)
    batch.ar231_state = state
    return state

def prepared_positions(batch):
    state = batch.ar231_state
    if state is not None and batch.forward_mode.is_decode() and batch.spec_info is None:
        if batch.seq_lens is state[0] and batch.orig_seq_lens is state[1] and batch.out_cache_loc is state[3]:
            return state[2]
    return None

def readback():
    errors = 0 if ERRORS is None else int(ERRORS[0])
    assert errors == 0, ('AR231 invalid decode preparation inputs',errors)
    return dict(enabled=ENABLED,calls=CALLS,pages=PAGES,errors=errors,fresh_outputs=True)
