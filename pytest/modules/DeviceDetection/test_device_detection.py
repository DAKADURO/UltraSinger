"""Tests for device_detection.py"""

import unittest

from src.modules.DeviceDetection.device_detection import recommend_whisper_settings


class RecommendWhisperSettingsTest(unittest.TestCase):
    def test_no_gpu_keeps_defaults(self):
        self.assertEqual(recommend_whisper_settings(None), (16, None))

    def test_big_gpu(self):
        self.assertEqual(recommend_whisper_settings(24), (16, None))
        self.assertEqual(recommend_whisper_settings(16), (16, None))

    def test_12gb_gpu(self):
        self.assertEqual(recommend_whisper_settings(12), (8, None))

    def test_8gb_gpu_uses_int8(self):
        self.assertEqual(recommend_whisper_settings(8), (8, "int8"))

    def test_small_gpu(self):
        self.assertEqual(recommend_whisper_settings(6), (4, "int8"))


if __name__ == "__main__":
    unittest.main()
