"""Tests for pitch_correction.py"""

import unittest

import numpy as np

from src.modules.Midi.pitch_correction import (
    allowed_pitch_classes,
    detect_key_from_pitches,
    fill_missing,
    fix_octaves,
    hz_to_midi,
    soft_quantize,
    steady_pitch,
)


class SteadyPitchTest(unittest.TestCase):
    def test_ignores_frames_outside_the_singing_range_and_low_confidence(self):
        freqs = [440.0, 441.0, 439.0, 1975.0, 440.0, 220.0]
        confs = [0.9, 0.9, 0.9, 0.95, 0.2, 0.9]

        pitch = steady_pitch(freqs, confs)

        self.assertAlmostEqual(pitch, 69.0, delta=0.1)

    def test_steady_part_wins_over_the_glide(self):
        # three frames glide up from G4, seven frames sit on A4
        freqs = [392.0, 410.0, 425.0] + [440.0] * 7
        confs = [0.9] * 10

        self.assertAlmostEqual(steady_pitch(freqs, confs), 69.0, delta=0.1)

    def test_nothing_usable(self):
        self.assertIsNone(steady_pitch([2000.0, 30.0], [0.9, 0.9]))
        self.assertIsNone(steady_pitch([440.0], [0.1]))


class FillMissingTest(unittest.TestCase):
    def test_uses_the_previous_then_the_following_note(self):
        self.assertEqual(fill_missing([None, 62.0, None, 64.0]), [62.0, 62.0, 62.0, 64.0])

    def test_all_missing(self):
        self.assertEqual(fill_missing([None, None], default=60.0), [60.0, 60.0])


class FixOctavesTest(unittest.TestCase):
    def test_moves_an_octave_error_towards_its_neighbours(self):
        pitches = [60.0, 62.0, 64.0, 74.0 + 12, 62.0, 60.0, 62.0]

        fixed = fix_octaves(pitches)

        self.assertAlmostEqual(fixed[3], 62.0)  # 86 -> 62 (two octaves)
        self.assertEqual(fixed[:3], pitches[:3])

    def test_a_real_melodic_leap_is_kept(self):
        pitches = [60.0, 62.0, 64.0, 67.0, 62.0, 60.0, 62.0]  # 67 is 3-5 semitones away from the median

        self.assertEqual(fix_octaves(pitches), pitches)

    def test_too_few_notes(self):
        self.assertEqual(fix_octaves([60.0, 90.0]), [60.0, 90.0])


class KeyTest(unittest.TestCase):
    def test_detects_c_major_from_its_scale(self):
        pitches = [60, 62, 64, 65, 67, 69, 71, 60, 64, 67, 60, 64, 67, 72]

        key = detect_key_from_pitches(pitches, [1.0] * len(pitches))

        self.assertEqual((key[0], key[1]), ("C", "major"))

    def test_no_notes(self):
        self.assertIsNone(detect_key_from_pitches([], []))

    def test_allowed_pitch_classes(self):
        self.assertEqual(allowed_pitch_classes("C", "major"), {0, 2, 4, 5, 7, 9, 11})
        self.assertEqual(allowed_pitch_classes("A", "minor"), {9, 11, 0, 2, 4, 5, 7})


class SoftQuantizeTest(unittest.TestCase):
    C_MAJOR = {0, 2, 4, 5, 7, 9, 11}

    def test_note_in_key_is_only_rounded(self):
        self.assertEqual(soft_quantize(64.2, self.C_MAJOR), 64)

    def test_clear_chromatic_note_stays(self):
        self.assertEqual(soft_quantize(61.05, self.C_MAJOR), 61)  # C#4, clearly not C or D

    def test_ambiguous_note_moves_into_the_key(self):
        self.assertEqual(soft_quantize(60.6, self.C_MAJOR), 60)  # between C4 (in key) and C#4
        self.assertEqual(soft_quantize(61.45, self.C_MAJOR), 62)  # between C#4 and D4 (in key)


class HzToMidiTest(unittest.TestCase):
    def test_a4(self):
        self.assertAlmostEqual(float(hz_to_midi(440.0)), 69.0)
        self.assertAlmostEqual(float(hz_to_midi(261.6256)), 60.0, places=3)


if __name__ == "__main__":
    unittest.main()
