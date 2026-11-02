# SPDX-License-Identifier: Apache-2.0
"""Encoded B70 expert geometry; each field keeps its checkpoint format."""

from sglang.srt.model_loader.gguf_native_reader import BLOCK_GEOMETRY, GGML_TYPES

GATE_UP_TYPES = frozenset((16, 21, 22, 23, 29, 42))
DOWN_TYPES = frozenset((8, 20, 42))


def expert_field_sizes(gate_up_kind, down_kind=42):
    if gate_up_kind not in GATE_UP_TYPES or down_kind not in DOWN_TYPES:
        raise ValueError("Unsupported B70 expert field formats")
    if down_kind != 42 and gate_up_kind not in (21, 23):
        raise ValueError("Mixed expert down formats require original IQ3_S/IQ4_XS gate/up")
    gu_block, gu_bytes = BLOCK_GEOMETRY[GGML_TYPES[gate_up_kind]]
    down_block, down_bytes = BLOCK_GEOMETRY[GGML_TYPES[down_kind]]
    return (
        640 * (2560 // gu_block) * gu_bytes,
        640 * (2560 // gu_block) * gu_bytes,
        2560 * (640 // down_block) * down_bytes,
    )
