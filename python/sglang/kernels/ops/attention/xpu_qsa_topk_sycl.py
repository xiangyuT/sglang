# SPDX-License-Identifier: Apache-2.0
"""Stable TopK512 of QSA scores with a zero-start valid prefix on B70.

The scorer must fill the invalid suffix with -inf. Short prefixes use one
workgroup per row; longer prefixes use tiled selection and pairwise merges.
Ordering follows Torch XPU float radix keys, including signed zero and NaN
payloads, then the original column index for ties. All lengths stay on device.
"""

import ctypes
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def load_library(path):
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError("Warm the QSA TopK library before graph capture")
    library = ctypes.CDLL(path)
    try:
        function = library.xpu_qsa_hybrid_top512
        error = library.xpu_qsa_hybrid_top512_error
    except AttributeError as exc:
        raise RuntimeError("Rebuild the QSA library for native TopK512") from exc
    function.argtypes = (
        [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2 + [ctypes.c_int64, ctypes.c_int]
    )
    function.restype = ctypes.c_int
    error.restype = ctypes.c_char_p
    return library


def supports(logits, lengths):
    return (
        logits.device.type == "xpu"
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and 1 <= logits.shape[0] <= 64
        and 1 <= logits.shape[1] <= 65536
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and lengths.device == logits.device
        and lengths.ndim == 1
        and lengths.shape[0] == logits.shape[0]
        and lengths.stride(0) == 1
        and lengths.dtype in (torch.int32, torch.int64)
        and torch.xpu.get_device_properties(logits.device).device_id == 0xE223
    )


def topk(logits, lengths, path):
    """Return int32 indices padded to 512 with -1, without host synchronization.

    Workspace belongs to this invocation and is stream-recorded. During capture
    it lives in the graph's pool, so simultaneous requests/graphs share no global
    scratch buffers. The native adapter requires an in-order SYCL queue.
    """
    if not supports(logits, lengths):
        raise ValueError("Unsupported QSA TopK512 geometry or device")
    library = load_library(path)
    rows, columns = logits.shape
    output = torch.empty((rows, 512), dtype=torch.int32, device=logits.device)
    work = []
    if columns > 2048:
        tiles = (columns + 1023) // 1024
        work = [
            torch.empty((rows, tiles, 512), dtype=torch.int64, device=logits.device)
            for _ in range(2)
        ]
    stream = torch.xpu.current_stream(logits.device)
    status = library.xpu_qsa_hybrid_top512(
        stream.sycl_queue,
        logits.data_ptr(),
        lengths.data_ptr(),
        output.data_ptr(),
        work[0].data_ptr() if work else None,
        work[1].data_ptr() if work else None,
        rows,
        columns,
        logits.stride(0),
        lengths.element_size(),
    )
    if status:
        message = library.xpu_qsa_hybrid_top512_error().decode()
        raise RuntimeError(f"Native QSA TopK512 failed: status={status}, {message}")
    for value in (logits, lengths, output, *work):
        value.record_stream(stream)
    return output
