"""Tests for bpm.py"""

import unittest

from src.modules.Audio.bpm import fold_tempo


class FoldTempoTest(unittest.TestCase):
    def test_tempo_in_range_is_unchanged(self):
        self.assertEqual(fold_tempo(120.0), 120.0)

    def test_double_tempo_is_halved(self):
        self.assertEqual(fold_tempo(240.0), 120.0)

    def test_half_tempo_is_doubled(self):
        self.assertEqual(fold_tempo(60.0), 120.0)

    def test_upper_bound_is_exclusive(self):
        self.assertEqual(fold_tempo(150.0), 75.0)

    def test_zero_is_returned_as_is(self):
        self.assertEqual(fold_tempo(0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
