"""Safe native-XPU reference for standard, local-expert EP dispatch.

The installed native prepare kernels require nonnegative local IDs. Compute
masked routes with local expert zero and a zero coefficient; no dummy weights
or copies of the expert parameters are needed. This retains native GEMMs and
is a development reference, not an EP decode optimization.
"""
import logging

import torch

logger = logging.getLogger(__name__)
_announced = False


def mask_nonlocal_routes(ids, weights):
    global _announced
    if not _announced:
        logger.info("XPU EP native safe routes enabled: nonlocal ID -> 0, coefficient -> 0")
        _announced = True
    local = ids >= 0
    return torch.where(local, ids, 0), torch.where(local, weights, 0.0)
