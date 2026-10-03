"""Tests for ffmpeg_helper.py"""

import subprocess
import unittest
from unittest.mock import patch

from src.modules.ffmpeg_helper import is_video_file


def _ffprobe_result(stdout: str, returncode: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


@patch("src.modules.ffmpeg_helper.get_ffmpeg_and_ffprobe_paths", return_value=("ffmpeg", "ffprobe"))
class IsVideoFileTest(unittest.TestCase):
    @patch("src.modules.ffmpeg_helper.subprocess.run")
    def test_real_video_stream(self, mock_run, _paths):
        mock_run.return_value = _ffprobe_result("0\n")

        self.assertTrue(is_video_file("clip.mp4"))

    @patch("src.modules.ffmpeg_helper.subprocess.run")
    def test_mp3_with_cover_art_is_not_a_video(self, mock_run, _paths):
        mock_run.return_value = _ffprobe_result("1\n")

        self.assertFalse(is_video_file("song.mp3"))

    @patch("src.modules.ffmpeg_helper.subprocess.run")
    def test_audio_only_file(self, mock_run, _paths):
        mock_run.return_value = _ffprobe_result("")

        self.assertFalse(is_video_file("song.wav"))

    @patch("src.modules.ffmpeg_helper.subprocess.run")
    def test_ffprobe_failure(self, mock_run, _paths):
        mock_run.return_value = _ffprobe_result("0\n", returncode=1)

        self.assertFalse(is_video_file("broken.mp4"))


if __name__ == "__main__":
    unittest.main()
