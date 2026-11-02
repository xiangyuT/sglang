# SPDX-License-Identifier: Apache-2.0
"""Select normal GGUF prefill by execution mode, including short cache tails."""


def is_gguf_prefill():
    from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch

    try:
        mode = get_forward_batch().forward_mode
    except RuntimeError:
        return False
    return mode.is_extend_without_speculative()
