"""Reject incompatible native windows before touching device storage."""

import unittest
from unittest.mock import Mock

from sglang.kernels.ops.speculative.dflash_native import (
    native_window,
    require_native_window,
)


class TestDflashNativeWindow(unittest.TestCase):
    def test_legacy_library_is_m4(self):
        library = object()
        self.assertEqual(native_window(library), 4)
        require_native_window(library, 4)
        with self.assertRaisesRegex(RuntimeError, "M4 cannot run M8"):
            require_native_window(library, 8)

    def test_explicit_windows_and_both_mismatches(self):
        for window, other in ((4, 8), (8, 4)):
            library = Mock(spec=["dflash_native_window"])
            library.dflash_native_window.return_value = window
            self.assertEqual(native_window(library), window)
            require_native_window(library, window)
            with self.assertRaisesRegex(RuntimeError, f"M{window} cannot run M{other}"):
                require_native_window(library, other)
            library.dflash_native_window.assert_called_once()

    def test_invalid_library_window_is_explicit(self):
        library = Mock(spec=["dflash_native_window"])
        library.dflash_native_window.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "Unsupported.*window: 7"):
            native_window(library)


if __name__ == "__main__":
    unittest.main()
