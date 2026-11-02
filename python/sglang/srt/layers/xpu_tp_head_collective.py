"""Graphable plain-greedy head sidebands over the existing B70 TP2 peer AR.

Each rank writes its packet bytes as exact BF16 integers in disjoint carrier
positions. The peer sum adds only zero; the original FP32 threshold bits and
BF16 score/token keys survive without arithmetic on their encoded values.
"""

import ctypes as C
import hashlib
import logging
from pathlib import Path

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class HeadCollective:
    def __init__(self, peer):
        if peer is None or peer.world != 2:
            raise ValueError('TP head graph collectives require the local TP2 peer AR')
        if torch.xpu.is_current_stream_capturing():
            raise RuntimeError('Warm TP head graph collectives before capture')
        self.peer = peer
        self.device = peer.device
        path = Path(envs.SGLANG_XPU_TP_HEAD_GRAPH_COLLECTIVES_LIB.get())
        if not path.is_file():
            raise RuntimeError('Set SGLANG_XPU_TP_HEAD_GRAPH_COLLECTIVES_LIB to the codec library')
        self.library_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.lib = C.CDLL(str(path))
        self.lib.head_codec_error.restype = C.c_char_p
        self.lib.head_codec_max_pack.argtypes = [C.c_void_p] * 3 + [C.c_int]
        self.lib.head_codec_max_finish.argtypes = [C.c_void_p] * 3
        self.lib.head_codec_prepare.argtypes = [C.c_void_p] * 4 + [C.c_int, C.c_int]
        self.lib.head_codec_scatter.argtypes = [C.c_void_p] * 3 + [C.c_int]
        self.partial = torch.empty(122, dtype=torch.int64, device=self.device)
        self.carrier = torch.empty((1, 2048), dtype=torch.bfloat16, device=self.device)
        if not peer.can_use(self.carrier):
            raise ValueError('Unsupported peer carrier')
        logger.info(
            'B70 TP head graph collectives enabled: rank=%d threshold FP32 bits, '
            'BF16 greedy first-token keys, carrier=4096B, peer=%s codec=%s',
            peer.rank, peer.library_sha256, self.library_sha256,
        )

    def check(self, rc):
        if rc:
            raise RuntimeError(self.lib.head_codec_error().decode())

    @property
    def queue(self):
        return torch.xpu.current_stream(self.device).sycl_queue

    def maximum(self, threshold):
        if not (threshold.device == self.device and threshold.dtype == torch.float32
                and threshold.numel() == 1 and threshold.is_contiguous()):
            raise ValueError('Expected one contiguous XPU FP32 threshold')
        self.check(self.lib.head_codec_max_pack(
            self.queue, threshold.data_ptr(), self.carrier.data_ptr(), self.peer.rank))
        self.peer.all_reduce(self.carrier)
        self.check(self.lib.head_codec_max_finish(
            self.queue, self.carrier.data_ptr(), threshold.data_ptr()))
        stream = torch.xpu.current_stream(self.device)
        for tensor in [threshold, self.carrier]:
            tensor.record_stream(stream)

    def gather_greedy(self, logits):
        if not (logits.device == self.device and logits.dtype == torch.bfloat16
                and logits.shape == (1, 124160) and logits.is_contiguous()):
            raise ValueError('Expected one contiguous BF16 TP2 head shard')
        out = torch.empty((1, 248320), dtype=torch.bfloat16, device=self.device)
        self.check(self.lib.head_codec_prepare(
            self.queue, logits.data_ptr(), self.partial.data_ptr(),
            self.carrier.data_ptr(), 124160, self.peer.rank))
        self.peer.all_reduce(self.carrier)
        self.check(self.lib.head_codec_scatter(
            self.queue, self.carrier.data_ptr(), out.data_ptr(), 248320))
        stream = torch.xpu.current_stream(self.device)
        for tensor in [logits, out, self.partial, self.carrier]:
            tensor.record_stream(stream)
        return out
