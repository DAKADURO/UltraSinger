"""Tests for audio_tags.py"""

import unittest
from unittest.mock import patch

from src.modules.audio_tags import read_audio_tags


class ReadAudioTagsTest(unittest.TestCase):
    @patch("src.modules.audio_tags.mutagen.File")
    def test_reads_artist_title_and_year(self, mock_file):
        mock_file.return_value = {"artist": ["Nacho Vegas"], "title": ["Mi canción"], "date": ["2006-09-26"]}

        tags = read_audio_tags("song.mp3")

        self.assertEqual(tags.artist, "Nacho Vegas")
        self.assertEqual(tags.title, "Mi canción")
        self.assertEqual(tags.year, "2006")

    @patch("src.modules.audio_tags.mutagen.File")
    def test_original_date_wins_over_date(self, mock_file):
        mock_file.return_value = {"originaldate": ["1999"], "date": ["2015-01-01"]}

        self.assertEqual(read_audio_tags("song.mp3").year, "1999")

    @patch("src.modules.audio_tags.mutagen.File")
    def test_year_without_four_digits_is_ignored(self, mock_file):
        mock_file.return_value = {"date": ["unknown"]}

        self.assertIsNone(read_audio_tags("song.mp3").year)

    @patch("src.modules.audio_tags.mutagen.File")
    def test_file_without_tags(self, mock_file):
        mock_file.return_value = None

        tags = read_audio_tags("song.mp3")

        self.assertEqual((tags.artist, tags.title, tags.year), (None, None, None))

    @patch("src.modules.audio_tags.mutagen.File", side_effect=OSError("broken"))
    def test_unreadable_file_gives_empty_tags(self, _mock_file):
        tags = read_audio_tags("song.mp3")

        self.assertEqual((tags.artist, tags.title, tags.year), (None, None, None))


if __name__ == "__main__":
    unittest.main()
