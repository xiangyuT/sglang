"""Exact E4M3 decoding with power-of-two compensation in group scales."""

import triton
import triton.language as tl


@triton.jit
def fp8_e4m3_div_256(value):
    # E4M3's bit pattern embeds FP8/256 exactly in FP16, including subnormals.
    # Move the compensating power of two to the per-group FP16-rounded scale.
    bits = value.to(tl.uint8, bitcast=True).to(tl.uint16)
    magnitude = bits & 127
    sign = (bits & 128) << 8
    half_bits = (magnitude << 7) | sign
    half_bits = tl.where(magnitude == 127, sign | 0x7E00, half_bits)
    return half_bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
