"""Tests for lyrics.py"""

import unittest

from src.modules.Speech_Recognition.lyrics import map_lyrics_to_segments, parse_lyrics


class ParseLyricsTest(unittest.TestCase):
    def test_removes_section_headers_and_blank_lines(self):
        text = "[Verse 1]\nHello world\n\n(Chorus)\nSing it loud\n"
        self.assertEqual(parse_lyrics(text), ["Hello", "world", "Sing", "it", "loud"])


class MapLyricsToSegmentsTest(unittest.TestCase):
    def test_replaces_misrecognised_words_keeping_segment_layout(self):
        asr = [["hello", "wold", "to"], ["night", "we", "sing"]]
        lyrics = "hello world tonight we sing".split()
        result, ratio = map_lyrics_to_segments(asr, lyrics)
        self.assertEqual(result, [["hello", "world"], ["tonight", "we", "sing"]])
        self.assertGreaterEqual(ratio, 0.5)

    def test_all_lyric_words_are_used_once(self):
        asr = [["la", "la"], ["la", "xx"], ["la", "la"]]
        lyrics = ["la", "la", "la", "lo", "la", "la"]
        result, _ = map_lyrics_to_segments(asr, lyrics)
        self.assertEqual([w for seg in result for w in seg], lyrics)

    def test_ignores_case_and_punctuation_when_matching(self):
        asr = [["Hello,", "World!"]]
        result, ratio = map_lyrics_to_segments(asr, ["hello", "world"])
        self.assertEqual(result, [["hello", "world"]])
        self.assertEqual(ratio, 1.0)

    def test_without_any_match_spreads_lyrics_evenly(self):
        asr = [["aaa", "bbb"], ["ccc", "ddd"]]
        lyrics = ["one", "two", "three", "four"]
        result, ratio = map_lyrics_to_segments(asr, lyrics)
        self.assertEqual(result, [["one", "two"], ["three", "four"]])
        self.assertEqual(ratio, 0.0)

    def test_missed_first_and_last_words_stay_in_first_and_last_segment(self):
        asr = [["two", "three"]]
        result, _ = map_lyrics_to_segments(asr, ["one", "two", "three", "four"])
        self.assertEqual(result, [["one", "two", "three", "four"]])

    def test_long_unrecognised_intro_is_not_forced_into_first_segment(self):
        lyrics = [f"w{i}" for i in range(10)] + ["two", "three"]
        result, _ = map_lyrics_to_segments([["two", "three"]], lyrics)
        self.assertEqual(result, [["two", "three"]])


if __name__ == "__main__":
    unittest.main()
