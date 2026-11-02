"""Window contract for the explicitly built DFlash expert/metadata/head library."""

import ctypes
from functools import lru_cache


@lru_cache(maxsize=None)
def native_window(library):
    query = getattr(library, "dflash_native_window", None)
    # Libraries built before M8 support have the retained M4 ABI.
    if query is None:
        return 4
    query.argtypes = []
    query.restype = ctypes.c_int
    window = query()
    if window not in (4, 8):
        raise RuntimeError(f"Unsupported DFlash native library window: {window}")
    return window


def require_native_window(library, window):
    actual = native_window(library)
    if actual != window:
        raise RuntimeError(
            f"DFlash native library window M{actual} cannot run M{window}; "
            "rebuild with --window-size and restart/recapture"
        )
