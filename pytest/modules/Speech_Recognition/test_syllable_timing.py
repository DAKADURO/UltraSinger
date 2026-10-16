"""Tests for syllable_timing.py"""

import unittest

from modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.Speech_Recognition.syllable_timing import (
    add_hyphen_to_data,
    assign_chars_to_words,
    syllable_times_from_chars,
)


def chars_of(text, starts):
    return [{"char": c, "start": s} for c, s in zip(text, starts)]


class AssignCharsToWordsTest(unittest.TestCase):
    def test_splits_flat_char_list_by_word(self):
        chars = [{"char": "h", "start": 0.0}, {"char": "i", "start": 0.1}, {"char": " "},
                 {"char": "y", "start": 0.3}, {"char": "o", "start": 0.4}]
        result = assign_chars_to_words(["hi", "yo"], chars)
        self.assertEqual([[c["char"] for c in w] for w in result], [["h", "i"], ["y", "o"]])

    def test_returns_none_when_chars_do_not_match(self):
        chars = [{"char": "x", "start": 0.0}, {"char": "y", "start": 0.1}]
        self.assertIsNone(assign_chars_to_words(["hi"], chars))

    def test_returns_none_when_chars_are_missing(self):
        self.assertIsNone(assign_chars_to_words(["hello"], [{"char": "h", "start": 0.0}]))

    def test_is_case_insensitive(self):
        result = assign_chars_to_words(["Hi"], [{"char": "h"}, {"char": "i"}])
        self.assertEqual(len(result[0]), 2)


class SyllableTimesTest(unittest.TestCase):
    def test_uses_char_start_of_next_syllable_as_boundary(self):
        # "cantar" -> can-tar, the long vowel makes the first syllable longer than half
        word_chars = chars_of("cantar", [1.0, 1.1, 1.9, 2.0, 2.1, 2.2])
        result = syllable_times_from_chars(1.0, 2.4, ["can", "tar"], word_chars)
        self.assertEqual(result, [(1.0, 2.0), (2.0, 2.4)])

    def test_syllables_are_contiguous_and_cover_the_word(self):
        word_chars = chars_of("canta", [0.0, 0.2, 0.4, 0.6, 0.8])
        result = syllable_times_from_chars(0.0, 1.0, ["can", "ta"], word_chars)
        self.assertEqual(result[0][0], 0.0)
        self.assertEqual(result[-1][1], 1.0)
        self.assertEqual(result[0][1], result[1][0])

    def test_untimed_chars_are_interpolated(self):
        word_chars = [{"char": "a", "start": 0.0}, {"char": "1"}, {"char": "b", "start": 1.0}]
        result = syllable_times_from_chars(0.0, 1.5, ["a1", "b"], word_chars)
        self.assertEqual(result, [(0.0, 1.0), (1.0, 1.5)])

    def test_inconsistent_char_count_returns_none(self):
        word_chars = chars_of("cant", [0.0, 0.2, 0.4, 0.6])
        self.assertIsNone(syllable_times_from_chars(0.0, 1.0, ["can", "ta"], word_chars))

    def test_too_short_syllable_returns_none(self):
        word_chars = chars_of("canta", [0.0, 0.01, 0.02, 0.03, 0.5])
        self.assertIsNone(syllable_times_from_chars(0.0, 1.0, ["can", "ta"], word_chars))

    def test_punctuation_in_syllables_is_counted(self):
        word_chars = chars_of("hola,", [0.0, 0.2, 0.4, 0.6, 0.7])
        result = syllable_times_from_chars(0.0, 1.0, ["ho", "la,"], word_chars)
        self.assertEqual(result, [(0.0, 0.4), (0.4, 1.0)])

    def test_single_syllable_returns_none(self):
        self.assertIsNone(syllable_times_from_chars(0.0, 1.0, ["la"], chars_of("la", [0.0, 0.5])))


class AddHyphenToDataTest(unittest.TestCase):
    def test_even_split_without_char_timings(self):
        data = [TranscribedData(word="cantar ", start=0.0, end=1.0)]
        result = add_hyphen_to_data(data, [["can", "tar "]])
        self.assertEqual([(d.word, d.start, d.end) for d in result],
                         [("can", 0.0, 0.5), ("tar ", 0.5, 1.0)])
        self.assertEqual([d.is_word_end for d in result], [False, True])
        self.assertTrue(all(d.is_hyphen for d in result))

    def test_uses_char_timings_when_available(self):
        data = [TranscribedData(word="cantar ", start=1.0, end=2.4,
                                char_timings=chars_of("cantar", [1.0, 1.1, 1.9, 2.0, 2.1, 2.2]))]
        result = add_hyphen_to_data(data, [["can", "tar "]])
        self.assertEqual([(d.start, d.end) for d in result], [(1.0, 2.0), (2.0, 2.4)])

    def test_words_without_hyphenation_are_kept(self):
        data = [TranscribedData(word="la ", start=0.0, end=1.0)]
        self.assertEqual(add_hyphen_to_data(data, [None]), data)

    def test_falls_back_to_even_split_on_inconsistent_char_timings(self):
        data = [TranscribedData(word="cantar ", start=0.0, end=1.0, char_timings=chars_of("xx", [0.0, 0.1]))]
        result = add_hyphen_to_data(data, [["can", "tar "]])
        self.assertEqual([(d.start, d.end) for d in result], [(0.0, 0.5), (0.5, 1.0)])


if __name__ == "__main__":
    unittest.main()
