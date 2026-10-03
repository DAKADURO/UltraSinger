"""Tests for os_helper.py"""

import unittest

from src.modules.os_helper import sanitize_filename


class SanitizeFilenameTest(unittest.TestCase):
    def test_characters_that_are_not_allowed_in_windows_names(self):
        self.assertEqual(sanitize_filename("Ha*Ash - Ex de verdad (En vivo)"), "Ha-Ash - Ex de verdad (En vivo)")
        self.assertEqual(sanitize_filename('AC/DC - Who? "Made": <Of> Me|'), "AC-DC - Who Made (Of) Me-")

    def test_trailing_dots_are_removed(self):
        self.assertEqual(sanitize_filename("Dry Martini, S. A."), "Dry Martini, S. A")

    def test_normal_names_are_kept(self):
        self.assertEqual(sanitize_filename("Maná - Labios compartidos"), "Maná - Labios compartidos")


if __name__ == "__main__":
    unittest.main()
