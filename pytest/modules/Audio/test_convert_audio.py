"""Tests for convert_audio.py"""

import subprocess
import unittest
from unittest.mock import patch

from src.modules.Audio.convert_audio import convert_audio_format


def _result(returncode: int, stderr: str = ""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


class ConvertAudioFormatTest(unittest.TestCase):
    @patch("src.modules.Audio.convert_audio.subprocess.run")
    def test_success_runs_ffmpeg_once_with_vbr(self, mock_run):
        mock_run.return_value = _result(0)

        convert_audio_format("in.wav", "out.mp3")

        self.assertEqual(mock_run.call_count, 1)
        self.assertIn("-q:a", mock_run.call_args.args[0])

    @patch("src.modules.Audio.convert_audio.subprocess.run")
    def test_mp3_retries_with_constant_bitrate(self, mock_run):
        mock_run.side_effect = [_result(1, "lame assertion"), _result(0)]

        convert_audio_format("in.wav", "out.mp3")

        retry_command = mock_run.call_args_list[1].args[0]
        self.assertNotIn("-q:a", retry_command)
        self.assertEqual(retry_command[retry_command.index("-b:a") + 1], "320k")

    @patch("src.modules.Audio.convert_audio.subprocess.run")
    def test_error_when_retry_fails_too(self, mock_run):
        mock_run.side_effect = [_result(1, "first"), _result(1, "second")]

        with self.assertRaises(RuntimeError):
            convert_audio_format("in.wav", "out.mp3")

    @patch("src.modules.Audio.convert_audio.subprocess.run")
    def test_other_formats_are_not_retried(self, mock_run):
        mock_run.return_value = _result(1, "boom")

        with self.assertRaises(RuntimeError):
            convert_audio_format("in.wav", "out.ogg")

        self.assertEqual(mock_run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
