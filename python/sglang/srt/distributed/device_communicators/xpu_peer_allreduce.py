"""Local B70 IPC BF16 collective with rank-ordered FP32 accumulation."""
import array
import ctypes as C
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import socket

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


class PeerAllReduce:
    def __init__(self, cpu_group, library):
        self.group = cpu_group
        self.rank = dist.get_rank(cpu_group)
        self.world = dist.get_world_size(cpu_group)
        assert self.world in (2, 4)
        self.device = torch.device('xpu', torch.xpu.current_device())
        assert torch.xpu.get_device_properties(self.device).device_id == 0xe223
        self.lib = lib = C.CDLL(library)
        self.library_sha256 = hashlib.sha256(Path(library).read_bytes()).hexdigest()
        lib.peer_ar_error.restype = C.c_char_p
        lib.peer_ar_create.argtypes = [C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_void_p)]
        lib.peer_ar_export.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_int), C.POINTER(C.c_uint64)]
        lib.peer_ar_open.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_uint32), C.c_int, C.c_uint64]
        lib.peer_ar_run.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
        self.variable_count = hasattr(lib, "peer_ar_run_n")
        self.max_elements = 65536 if self.variable_count else 2048
        if hasattr(lib, "peer_ar_max_elements"):
            lib.peer_ar_max_elements.restype = C.c_int
            self.max_elements = lib.peer_ar_max_elements()
        if self.variable_count:
            lib.peer_ar_run_n.argtypes = [C.c_void_p] * 3 + [C.c_int]
            lib.peer_ar_run_n.restype = C.c_int
        self.byte_allgather = hasattr(lib, "peer_ag_run")
        if self.byte_allgather:
            lib.peer_ag_run.argtypes = [C.c_void_p] * 4 + [C.c_int]
            lib.peer_ag_run.restype = C.c_int
            self.max_gather_bytes = 2097152
            if hasattr(lib, "peer_ag_max_bytes"):
                lib.peer_ag_max_bytes.restype = C.c_int
                self.max_gather_bytes = lib.peer_ag_max_bytes()
        lib.peer_ar_status.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_uint32)]
        lib.peer_ar_destroy.argtypes = [C.c_void_p, C.c_void_p]
        lib.peer_ar_poll.argtypes = [C.c_void_p]
        self.handle = C.c_void_p()
        self.check(lib.peer_ar_create(self.queue, self.rank, self.world, C.byref(self.handle)))
        words = (C.c_uint32 * 16)(); fd = C.c_int(); cookie = C.c_uint64()
        self.check(lib.peer_ar_export(self.handle, words, C.byref(fd), C.byref(cookie)))
        mine = json.dumps(dict(words=list(words), pid=os.getpid(), cookie=cookie.value,
                               rank=self.rank, host=socket.gethostname())).encode()
        hosts = [None] * self.world
        dist.all_gather_object(hosts, socket.gethostname(), group=cpu_group)
        assert len(set(hosts)) == 1, 'single host required'
        nonce = [secrets.token_hex(12) if self.rank == 0 else None]
        dist.broadcast_object_list(nonce, src=dist.get_global_rank(cpu_group, 0), group=cpu_group)
        addresses = ['\0sglang-b70-ar-'+nonce[0]+'-'+str(i) for i in range(self.world)]

        def exchange(conn, peer):
            conn.settimeout(120)
            conn.sendmsg([mine], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd.value]))])
            payload, anc, flags, _ = conn.recvmsg(8192, socket.CMSG_SPACE(4))
            assert not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
            received = array.array('i')
            for level, kind, data in anc:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    received.frombytes(data[:len(data)//4*4])
            assert len(received) == 1
            other = json.loads(payload)
            assert other['rank'] == peer and other['host'] == socket.gethostname()
            raw = (C.c_uint32 * 16)(*other['words'])
            try:
                self.check(lib.peer_ar_open(self.handle, self.queue, peer, received[0],
                                           raw, other['pid'], other['cookie']))
            except BaseException:
                os.close(received[0]); raise

        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
            listener.bind(addresses[self.rank]); listener.listen(self.world); listener.settimeout(120)
            dist.barrier(group=cpu_group)
            for lower in range(self.world):
                if self.rank > lower:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as conn:
                        conn.settimeout(120); conn.connect(addresses[lower])
                        conn.send(bytes([self.rank])); exchange(conn, lower)
                elif self.rank == lower:
                    for higher in range(lower+1, self.world):
                        # Peer ranks arrive in arbitrary order: get credentials
                        # from the explicit packet rather than accept order.
                        with listener.accept()[0] as conn:
                            conn.settimeout(120)
                            peer = conn.recv(1)[0]
                            exchange(conn, peer)
            dist.barrier(group=cpu_group)
        log.info('B70 peer all-reduce initialized: rank=%d world=%d library=%s FD=SCM_RIGHTS',
                 self.rank, self.world, self.library_sha256)

    @property
    def queue(self):
        return torch.xpu.current_stream(self.device).sycl_queue

    def check(self, rc):
        if rc:
            raise RuntimeError(self.lib.peer_ar_error().decode())

    def can_use(self, x):
        return (isinstance(x, torch.Tensor) and x.device == self.device
                and x.dtype == torch.bfloat16
                and (8 <= x.numel() and x.numel() % 8 == 0
                     and (x.numel() <= self.max_elements
                          or (os.environ.get("SGLANG_XPU_PEER_ALLREDUCE_CHUNKED") == "1"
                              and self.max_elements % 8 == 0
                              and not torch.xpu.is_current_stream_capturing()))
                     if self.variable_count else x.numel() == 2048)
                and x.is_contiguous() and x.data_ptr() % 16 == 0)

    def all_reduce(self, x):
        assert self.can_use(x)
        self.raise_if_failed()
        if self.variable_count:
            # The transport preserves its original per-element rank order.
            # Only eager tensors larger than its arena are split; decode and
            # captured calls retain the existing one-submission path.
            for first in range(0, x.numel(), self.max_elements):
                count = min(self.max_elements, x.numel() - first)
                self.check(self.lib.peer_ar_run_n(
                    self.handle, self.queue,
                    x.data_ptr() + first * x.element_size(), count,
                ))
        else:
            self.check(self.lib.peer_ar_run(self.handle, self.queue, x.data_ptr()))
        x.record_stream(torch.xpu.current_stream(self.device))
        return x

    def can_all_gather(self, output, input_):
        size = input_.numel() * input_.element_size()
        return (self.byte_allgather and input_.device == output.device == self.device
                and input_.dtype == output.dtype
                and 4 <= size and size % 4 == 0
                and (size <= self.max_gather_bytes or
                     os.environ.get("SGLANG_XPU_NATIVE_ALLGATHER_CHUNKED") == "1")
                and output.numel() == self.world * input_.numel()
                and input_.is_contiguous() and output.is_contiguous()
                and input_.data_ptr() % 4 == output.data_ptr() % 4 == 0)

    def all_gather(self, output, input_):
        assert self.can_all_gather(output, input_)
        self.raise_if_failed()
        if input_.numel() * input_.element_size() > self.max_gather_bytes:
            # Keep the IPC arena/ABI unchanged. Each rank executes the same
            # bounded collectives, then copies each rank's bytes to its final
            # output slice. Static slices and copies also capture in XPU graphs.
            flat = input_.view(-1)
            target = output.view(self.world, -1)
            capacity = self.max_gather_bytes // input_.element_size()
            for first in range(0, flat.numel(), capacity):
                part = flat[first:first + capacity]
                gathered = torch.empty((self.world, part.numel()), dtype=part.dtype, device=part.device)
                self.all_gather(gathered, part)
                target[:, first:first + part.numel()].copy_(gathered)
            return output
        self.check(self.lib.peer_ag_run(
            self.handle, self.queue, input_.data_ptr(), output.data_ptr(),
            input_.numel() * input_.element_size(),
        ))
        for value in (input_, output):
            value.record_stream(torch.xpu.current_stream(self.device))

    def raise_if_failed(self):
        if self.lib.peer_ar_poll(self.handle):
            raise RuntimeError('B70 peer all-reduce timed out; stop this communicator')

    def status(self):
        value = C.c_uint32()
        self.check(self.lib.peer_ar_status(self.handle, self.queue, C.byref(value)))
        return value.value

    def close(self):
        if self.handle:
            torch.xpu.synchronize(self.device)
            dist.barrier(group=self.group)
            self.check(self.lib.peer_ar_destroy(self.handle, self.queue))
            self.handle = C.c_void_p()
