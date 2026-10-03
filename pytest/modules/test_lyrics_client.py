"""Tests for lyrics_client.py"""

import unittest

import numpy as np

from src.modules.lyrics_client import (
    LyricLine,
    estimate_offset,
    lines_to_segments,
    parse_lrc,
    pick_best_result,
    trim_segment_ends,
)

LRC = """[ar:Someone]
[00:12.50] Primera línea
[00:15.00]
[01:02.5]Segunda línea
[00:30.123][01:30.000] Estribillo
no es una línea
"""


class ParseLrcTest(unittest.TestCase):
    def test_parses_sorted_lines_and_skips_tags_and_empty_lines(self):
        lines = parse_lrc(LRC)

        self.assertEqual([line.text for line in lines], ["Primera línea", "Estribillo", "Segunda línea", "Estribillo"])
        self.assertAlmostEqual(lines[0].start, 12.5)
        self.assertAlmostEqual(lines[1].start, 30.123)
        self.assertAlmostEqual(lines[2].start, 62.5)
        self.assertAlmostEqual(lines[3].start, 90.0)

    def test_empty_text(self):
        self.assertEqual(parse_lrc(""), [])


class PickBestResultTest(unittest.TestCase):
    RESULTS = [
        {"id": 1, "duration": 300, "syncedLyrics": "[00:01.00] live"},
        {"id": 2, "duration": 317, "syncedLyrics": "[00:01.00] studio"},
        {"id": 3, "duration": 316, "syncedLyrics": None, "plainLyrics": "no times"},
    ]

    def test_closest_duration_with_synced_lyrics(self):
        self.assertEqual(pick_best_result(self.RESULTS, 316.9)["id"], 2)

    def test_rejects_other_editions(self):
        self.assertIsNone(pick_best_result(self.RESULTS, 250))

    def test_without_duration_takes_first_synced(self):
        self.assertEqual(pick_best_result(self.RESULTS, None)["id"], 1)

    def test_no_synced_lyrics(self):
        self.assertIsNone(pick_best_result([{"id": 3, "duration": 1, "plainLyrics": "x"}], 1))


class EstimateOffsetTest(unittest.TestCase):
    LINES = [LyricLine(10.0, "Hola mundo azul"), LyricLine(20.0, "Otra línea distinta"), LyricLine(30.0, "Hola mundo azul")]

    def test_median_shift_of_matching_lines(self):
        segments = [
            {"text": "hola mundo azul", "start": 12.0},
            {"text": "otra linea distinta", "start": 22.1},
            {"text": "algo que no esta", "start": 40.0},
        ]

        self.assertAlmostEqual(estimate_offset(segments, self.LINES), 2.05, places=2)

    def test_repeated_line_matches_the_nearest_one(self):
        segments = [{"text": "hola mundo azul", "start": 31.0}]

        self.assertAlmostEqual(estimate_offset(segments, self.LINES), 1.0)

    def test_segment_with_several_lines_uses_the_start_of_the_first_line(self):
        lines = [
            LyricLine(2.0, "Es hora de recapitular"),
            LyricLine(10.3, "Las hostias que me ha dado el mundo"),
            LyricLine(18.2, "Hoy querrán oír mi último adiós"),
        ]
        segments = [{"text": "Es hora de recapitular las hostias que me ha dado el mundo hoy querrán oír mi último adiós", "start": 2.0}]

        self.assertAlmostEqual(estimate_offset(segments, lines), 0.0)

    def test_no_match_gives_no_shift(self):
        self.assertEqual(estimate_offset([{"text": "nada que ver", "start": 10.0}], self.LINES), 0.0)


class TrimSegmentEndsTest(unittest.TestCase):
    SAMPLE_RATE = 16000

    def _audio(self, voiced_from: float, voiced_until: float, seconds: float = 10.0):
        audio = np.zeros(int(seconds * self.SAMPLE_RATE))
        audio[int(voiced_from * self.SAMPLE_RATE):int(voiced_until * self.SAMPLE_RATE)] = 0.5
        return audio

    def test_last_word_ends_where_the_voice_stops(self):
        segments = [{"words": [{"word": "a", "start": 2.0, "end": 3.0}, {"word": "b", "start": 3.0, "end": 9.0}]}]

        trim_segment_ends(segments, self._audio(2.0, 4.0))

        self.assertAlmostEqual(segments[0]["words"][0]["end"], 3.0)
        self.assertAlmostEqual(segments[0]["words"][1]["end"], 4.05, places=2)

    def test_voice_of_the_next_line_is_not_part_of_the_word(self):
        audio = self._audio(2.0, 4.0)
        audio[int(9.5 * self.SAMPLE_RATE):int(10.0 * self.SAMPLE_RATE)] = 0.5
        segments = [{"words": [{"word": "a", "start": 2.0, "end": 9.9}]}]

        trim_segment_ends(segments, audio)

        self.assertAlmostEqual(segments[0]["words"][0]["end"], 4.05, places=2)

    def test_short_breath_inside_a_word_is_kept(self):
        audio = self._audio(2.0, 3.0)
        audio[int(3.3 * self.SAMPLE_RATE):int(4.0 * self.SAMPLE_RATE)] = 0.5
        segments = [{"words": [{"word": "a", "start": 2.0, "end": 9.0}]}]

        trim_segment_ends(segments, audio)

        self.assertAlmostEqual(segments[0]["words"][0]["end"], 4.05, places=2)

    def test_voice_until_the_end_keeps_the_word(self):
        segments = [{"words": [{"word": "a", "start": 2.0, "end": 5.0}]}]

        trim_segment_ends(segments, self._audio(2.0, 6.0))

        self.assertAlmostEqual(segments[0]["words"][0]["end"], 5.0)

    def test_words_without_timing_and_silence_are_ignored(self):
        segments = [{"words": [{"word": "1"}]}, {"words": [{"word": "a", "start": 7.0, "end": 8.0}]}]

        trim_segment_ends(segments, self._audio(2.0, 4.0))

        self.assertAlmostEqual(segments[1]["words"][0]["end"], 8.0)


class LinesToSegmentsTest(unittest.TestCase):
    def test_windows_follow_the_next_line_and_shift(self):
        lines = [LyricLine(10.0, "a"), LyricLine(20.0, "b")]

        segments = lines_to_segments(lines, offset=2.0, audio_duration=100.0)

        self.assertAlmostEqual(segments[0]["start"], 11.7)
        self.assertAlmostEqual(segments[0]["end"], 22.0)
        self.assertAlmostEqual(segments[1]["end"], 30.0)

    def test_end_is_limited_to_the_audio(self):
        segments = lines_to_segments([LyricLine(10.0, "a")], offset=0.0, audio_duration=12.0)

        self.assertAlmostEqual(segments[0]["end"], 12.0)

    def test_too_short_windows_are_dropped(self):
        segments = lines_to_segments([LyricLine(10.0, "a")], offset=0.0, audio_duration=10.1)

        self.assertEqual(segments, [])


if __name__ == "__main__":
    unittest.main()
