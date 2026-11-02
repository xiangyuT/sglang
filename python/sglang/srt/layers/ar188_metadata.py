"""Opt-in M=1 plain-AR metadata producer inside the existing full XPU graph.

Original eager metadata initializes Python objects before the first capture.
The native producer updates their existing tensor storage. Startup chooses one
graph mode; unsupported runtime shapes use the original eager forward path.
"""
import ctypes
import logging
import os
import torch

logger = logging.getLogger(__name__)
MODE = os.environ.get('CODEX_AR228_METADATA_MODE','original')
if MODE not in ['original','ingraph']:
    raise ValueError('AR228 metadata mode must be original or ingraph at startup')
LAST = None
RUNNER = None
GRAPH_MODE = 'original'
CAPTURES = []
BASE_NAMES = None
SWITCHES = 0

def enabled():
    return os.environ.get('CODEX_AR188_METADATA') == '1'

def _eligible(owner, fb, in_capture=False):
    full, linear = owner.full_attn_backend, owner.linear_attn_backend
    actual = getattr(fb,'actual_forward_mode',None)
    if actual is None:
        actual = fb.forward_mode
    return (
        enabled() and fb.batch_size == 1 and fb.forward_mode.is_decode()
        and actual.is_decode() and fb.spec_info is None
        and type(full).__name__ == 'XPUAttentionBackend'
        and type(linear).__name__ == 'XpuGDNAttnBackend'
        and full.page_size == 128 and not full.use_mla
        and not full.use_sliding_window_kv_pool and not full.is_encoder_decoder
        and full.attention_chunk_size is None and full.dcp_size == 1
        and linear.replayssm_write_pos_list is None
        and linear.req_to_token_pool.mamba_translate_is_fusable
        and (getattr(fb,'num_padding',0)==0 or (in_capture and getattr(fb,'num_padding',None) is None))
        and fb.seq_lens_cpu is not None
        and getattr(fb,'mamba_track_mask',None) is None
        and getattr(fb,'mamba_track_indices',None) is None
    )

def _pointer(tensor):
    return tensor.data_ptr() if tensor is not None else 0

class Context:
    def __init__(self, owner, fb):
        self.owner = owner
        self.full, self.linear = owner.full_attn_backend, owner.linear_attn_backend
        self.full_metadata = self.full.forward_metadata
        self.linear_metadata = self.linear.forward_metadata
        self.req, self.lens = fb.req_pool_indices, fb.seq_lens
        self.track = getattr(fb, 'mamba_track_indices', None)
        self.tokens = self.full.req_to_token
        self.mapping = self.linear.req_to_token_pool.req_index_to_mamba_index_mapping
        self.v2p = self.linear.req_to_token_pool.mamba_v2p_table
        self.track_out = self.linear_metadata.mamba_track_indices
        self.cache_len = self.full_metadata.cache_seqlens_int32
        self.cu_k = self.full_metadata.cu_seqlens_k
        self.pages = self.full_metadata.page_table
        self.state = self.linear_metadata.mamba_cache_indices
        self.query = self.linear_metadata.query_start_loc
        self.errors = torch.zeros(1, dtype=torch.int32, device=self.req.device)
        self.tensors = [self.req,self.lens,self.tokens,self.mapping,self.v2p,self.track,
                        self.cache_len,self.cu_k,self.pages,self.state,self.query,self.track_out,self.errors]
        for tensor in self.tensors:
            if tensor is not None:
                assert tensor.device == self.req.device and tensor.is_contiguous()
        assert self.req.device.type == 'xpu' and self.req.shape == self.lens.shape == (1,)
        assert self.req.dtype in (torch.int32,torch.int64) and self.lens.dtype in (torch.int32,torch.int64)
        assert self.tokens.dtype in (torch.int32,torch.int64) and self.tokens.ndim == 2
        assert self.mapping.shape == (self.tokens.shape[0],)
        assert all(t.dtype == torch.int32 for t in [self.mapping,self.cache_len,self.cu_k,self.pages,self.state,self.query])
        assert self.cache_len.shape == self.state.shape == (1,) and self.cu_k.shape == self.query.shape == (2,)
        assert self.pages.shape[0] == 1
        assert self.v2p is None or self.v2p.dtype == torch.int32
        assert (self.track is None) == (self.track_out is None)
        assert self.track is None or self.track.shape == self.track_out.shape == (1,)
        assert self.track is None or self.track.dtype == self.track_out.dtype == torch.int64
        assert torch.xpu.get_device_properties(self.req.device).device_id == 0xe223
        self.ints = [self.tokens.shape[0],self.tokens.shape[1],self.full.max_context_len,
            self.pages.shape[1],self.full.page_size,self.v2p.numel() if self.v2p is not None else 0,
            int(self.req.dtype == torch.int64),int(self.lens.dtype == torch.int64),
            int(self.tokens.dtype == torch.int64),self.linear.req_to_token_pool.mamba_pool.size+1]
        self.lib = ctypes.CDLL('/evidence/tpot-iterations-20261023/native/metadata194/metadata.so')
        self.lib.metadata188_run.argtypes = [ctypes.c_void_p]*14+[ctypes.c_int]*10
        self.lib.metadata188_error.restype = ctypes.c_char_p
        self.pointers = [_pointer(t) for t in self.tensors]
        self.identity = self._identity(fb)
        self.prepared_fb = None

    def _identity(self, fb):
        return tuple(_pointer(t) for t in [fb.req_pool_indices,fb.seq_lens,
            getattr(fb,'mamba_track_indices',None),self.full.req_to_token,
            self.linear.req_to_token_pool.req_index_to_mamba_index_mapping,
            self.linear.req_to_token_pool.mamba_v2p_table])

    def matches(self, fb):
        return self._identity(fb) == self.identity

    def bind(self, fb):
        self.full.forward_metadata = self.full_metadata
        self.linear.forward_metadata = self.linear_metadata
        # Keep the Python diagnostic field live; graph device work does not
        # freeze a launch range from this changing host value.
        self.full_metadata.max_seq_len_k = int(fb.seq_lens_cpu[0])
        self.prepared_fb = fb

    def run(self):
        stream = torch.xpu.current_stream(self.req.device)
        rc = self.lib.metadata188_run(stream.sycl_queue,*self.pointers,*self.ints)
        if rc:
            raise RuntimeError(self.lib.metadata188_error().decode())
        for tensor in self.tensors:
            if tensor is not None:
                tensor.record_stream(stream)

def remember(owner, fb):
    global LAST
    owner._meta188_ctx = None
    owner._meta188_active = False
    if not _eligible(owner,fb,in_capture=True):
        return
    owner._meta188_ctx = Context(owner,fb)
    owner._meta188_active = False
    LAST = owner._meta188_ctx
    logger.info('AR188 metadata ready: page=%d, req=%s, lens=%s, table=%s, v2p=%s, track=%s',
        owner.full_attn_backend.page_size, fb.req_pool_indices.dtype,fb.seq_lens.dtype,
        owner.full_attn_backend.req_to_token.dtype,LAST.v2p is not None,LAST.track is not None)

def try_fast_out(owner, fb):
    owner._meta188_active = False
    ctx = getattr(owner,'_meta188_ctx',None)
    if ctx is not None:
        ctx.prepared_fb = None
    if MODE not in ['eager','ingraph'] or ctx is None or not _eligible(owner,fb):
        return False
    if not ctx.matches(fb):
        if MODE == 'ingraph':
            raise AssertionError('AR229 metadata load changed captured input storage')
        return False
    ctx.bind(fb)
    if MODE == 'eager':
        ctx.run()
    owner._meta188_active = True
    return True

def run_in_graph(owner, fb):
    ctx = getattr(owner,'_meta188_ctx',None)
    if MODE == 'ingraph' and ctx is not None and _eligible(owner,fb,in_capture=True) and ctx.matches(fb):
        ctx.run()
        return True
    return False

def readback():
    if LAST is None:
        return dict(enabled=False,mode=MODE)
    errors = int(LAST.errors[0])
    assert errors == 0, ('AR188 invalid runtime metadata',errors)
    return dict(enabled=True,mode=MODE,graph_mode=GRAPH_MODE,captures=len(CAPTURES),captured_kernels=CAPTURES[-1]['kernels'] if CAPTURES else None,active=getattr(LAST.owner,'_meta188_active',False),
                errors=errors,request_rows=LAST.tokens.shape[0],token_stride=LAST.tokens.shape[1],
                page_columns=LAST.pages.shape[1],page_size=LAST.full.page_size,
                v2p=LAST.v2p is not None,track=LAST.track is not None)


def record_capture(path):
    import re
    global GRAPH_MODE, BASE_NAMES
    GRAPH_MODE = 'ingraph' if MODE == 'ingraph' else 'original'
    names = None
    if path is not None:
        names = re.findall(r'kernelName</TD>[^\n]*?<FONT FACE="monospace">(.*?)</FONT>',path.read_text())
        if BASE_NAMES is None:
            if GRAPH_MODE == 'ingraph':
                assert 'metadata194' in names[0]
                BASE_NAMES = names[1:]
            else:
                BASE_NAMES = names
        elif GRAPH_MODE == 'ingraph':
            assert 'metadata194' in names[0] and names[1:] == BASE_NAMES
        else:
            assert names == BASE_NAMES
    CAPTURES.append(dict(mode=GRAPH_MODE,path=str(path),kernels=None if names is None else len(names)))


def can_use_graph(fb):
    if not enabled() or MODE != 'ingraph':
        return True
    return LAST is not None and _eligible(LAST.owner,fb,in_capture=True)

def validate_graph_inputs(fb):
    if enabled() and MODE == 'ingraph':
        assert LAST is not None and _eligible(LAST.owner,fb,in_capture=True)
        # DecodeCudaGraphRunner.execute passes the live request batch here.
        # load_batch prepared a separate view over graph-owned input buffers.
        # Validate that view, and consume its preparation for this replay.
        assert LAST.prepared_fb is not None and LAST.owner._meta188_active, 'AR229 replay without static metadata preparation'
        assert LAST.matches(LAST.prepared_fb), 'AR229 prepared static graph storage changed'
        LAST.prepared_fb = None

def switch_mode(mode):
    # Cold graph mode is immutable; no second graph/native recapture is needed.
    assert mode == MODE, 'AR228 metadata graph mode is fixed at startup; restart to change'
