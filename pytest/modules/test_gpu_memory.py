"""Tests for gpu_memory.py"""

import unittest

from src.modules.gpu_memory import recommend_whisper_settings, resolve_whisper_settings


class WhisperSettingsTest(unittest.TestCase):
    def test_8gb_gpu_gets_quantized_model_and_small_batch(self):
        batch, compute = recommend_whisper_settings("cuda", 8.0)
        self.assertEqual(compute, "int8_float16")
        self.assertLessEqual(batch, 8)

    def test_big_gpu_keeps_float16(self):
        self.assertEqual(recommend_whisper_settings("cuda", 24.0), (16, "float16"))

    def test_cpu_uses_int8(self):
        self.assertEqual(recommend_whisper_settings("cpu", None)[1], "int8")

    def test_user_choices_win(self):
        self.assertEqual(resolve_whisper_settings("cuda", 3, "float16", vram_gb=8.0), (3, "float16"))

    def test_open_choices_are_filled(self):
        self.assertEqual(resolve_whisper_settings("cuda", None, "float32", vram_gb=8.0)[1], "float32")


if __name__ == "__main__":
    unittest.main()
