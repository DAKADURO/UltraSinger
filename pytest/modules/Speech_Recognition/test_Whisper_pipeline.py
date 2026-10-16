"""Tests for the chunked / memory friendly flow of Whisper.py, with whisperx and torch replaced by fakes"""

import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


class FakeOutOfMemory(Exception):
    pass


def build_fakes(events, batch_limit=None):
    """Fake torch / whisperx / num2words modules that record what the pipeline does"""
    torch = types.ModuleType("torch")
    torch.load = lambda *a, **k: None
    torch.cuda = types.ModuleType("torch.cuda")
    torch.cuda.OutOfMemoryError = FakeOutOfMemory
    torch.cuda.is_available = lambda: False
    torch.cuda.empty_cache = lambda: None

    class FakeModel:
        def __init__(self, device, compute_type):
            self.device, self.compute_type = device, compute_type

        def transcribe(self, audio, batch_size, language):
            events.append(("transcribe", batch_size, len(audio)))
            if batch_limit is not None and batch_size > batch_limit:
                raise RuntimeError("CUDA failed with error out of memory")
            return {"language": language or "es",
                    "segments": [{"text": "hola mundo", "start": 1.0, "end": 3.0}]}

    whisperx = types.ModuleType("whisperx")

    def load_model(name, language, device, compute_type):
        events.append(("load_model", device, compute_type))
        return FakeModel(device, compute_type)

    def load_align_model(language_code, device, model_name):
        events.append(("load_align", device))
        return object(), {}

    def align(segments, model_a, metadata, audio, device, return_char_alignments):
        events.append(("align", return_char_alignments, [s["text"] for s in segments]))
        out = []
        for seg in segments:
            words = seg["text"].split()
            out.append({"words": [{"word": w, "start": seg["start"] + i, "end": seg["start"] + i + 1, "score": 0.9}
                                  for i, w in enumerate(words)]})
        return {"segments": out}

    whisperx.load_model = load_model
    whisperx.load_align_model = load_align_model
    whisperx.align = align
    whisperx.load_audio = lambda path: np.zeros(16000 * 100, dtype=np.float32)

    num2words = types.ModuleType("num2words")
    num2words.num2words = lambda n, lang="en": str(n)
    return {"torch": torch, "torch.cuda": torch.cuda, "whisperx": whisperx, "num2words": num2words}


class WhisperPipelineTest(unittest.TestCase):
    def setUp(self):
        self.events = []

    def import_whisper(self, batch_limit=None):
        patcher = patch.dict(sys.modules, build_fakes(self.events, batch_limit))
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("modules.Speech_Recognition.Whisper", None)
        module = importlib.import_module("modules.Speech_Recognition.Whisper")
        module._get_silences = lambda path: [(50.0, 51.0)]
        return module

    def test_cuts_audio_in_chunks_and_offsets_timestamps(self):
        whisper = self.import_whisper()
        result = whisper.transcribe_with_whisper("x.wav", whisper.WhisperModel.TINY, device="cpu",
                                                 batch_size=4, compute_type="int8", chunk_seconds=60)
        transcribe_calls = [e for e in self.events if e[0] == "transcribe"]
        self.assertEqual(len(transcribe_calls), 2)
        starts = [d.start for d in result.transcribed_data]
        # chunk 1 starts at 0 s, chunk 2 at the silence middle 50.5 s
        self.assertEqual(starts, [1.0, 2.0, 51.5, 52.5])
        self.assertEqual(result.detected_language, "es")

    def test_whisper_is_unloaded_before_alignment_and_chars_are_requested(self):
        whisper = self.import_whisper()
        whisper.transcribe_with_whisper("x.wav", whisper.WhisperModel.TINY, device="cpu",
                                        batch_size=4, compute_type="int8")
        names = [e[0] for e in self.events]
        self.assertLess(names.index("transcribe"), names.index("load_align"))
        align_event = next(e for e in self.events if e[0] == "align")
        self.assertTrue(align_event[1])

    def test_out_of_memory_halves_batch_size_until_it_fits(self):
        whisper = self.import_whisper(batch_limit=2)
        whisper.transcribe_with_whisper("x.wav", whisper.WhisperModel.TINY, device="cpu",
                                        batch_size=16, compute_type="int8", chunk_seconds=0)
        batches = [e[1] for e in self.events if e[0] == "transcribe"]
        self.assertEqual(batches, [16, 8, 4, 2])

    def test_out_of_memory_on_cuda_falls_back_to_int8_float16_then_cpu(self):
        whisper = self.import_whisper(batch_limit=0)
        with self.assertRaises(Exception):
            whisper.transcribe_with_whisper("x.wav", whisper.WhisperModel.TINY, device="cuda",
                                            batch_size=1, compute_type="float16", chunk_seconds=0)
        loads = [e[1:] for e in self.events if e[0] == "load_model"]
        self.assertEqual(loads, [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")])

    def test_finished_chunks_are_reused_from_checkpoint(self):
        whisper = self.import_whisper()
        with tempfile.TemporaryDirectory() as folder:
            kwargs = dict(device="cpu", batch_size=4, compute_type="int8", checkpoint_dir=folder)
            whisper.transcribe_with_whisper("x.wav", whisper.WhisperModel.TINY, **kwargs)
            first_run = len([e for e in self.events if e[0] == "transcribe"])
            whisper.transcribe_with_whisper("x.wav", whisper.WhisperModel.TINY, **kwargs)
            second_run = len([e for e in self.events if e[0] == "transcribe"]) - first_run
            self.assertEqual(first_run, 2)
            self.assertEqual(second_run, 0)
            self.assertEqual(len(os.listdir(folder)), 2)

    def test_given_lyrics_replace_recognised_text(self):
        whisper = self.import_whisper()
        whisper.transcribe_with_whisper("x.wav", whisper.WhisperModel.TINY, device="cpu", batch_size=4,
                                        compute_type="int8", chunk_seconds=0,
                                        lyrics_words=["hola", "mundo", "cruel"])
        align_event = next(e for e in self.events if e[0] == "align")
        self.assertEqual(align_event[2], ["hola mundo cruel"])

    def test_char_timings_are_attached_to_words(self):
        whisper = self.import_whisper()
        result_aligned = {"segments": [{
            "words": [{"word": "hola", "start": 0.0, "end": 1.0, "score": 0.9}],
            "chars": [{"char": "h", "start": 0.0, "end": 0.1, "score": 0.9}, {"char": "o", "start": 0.2},
                      {"char": "l", "start": 0.5}, {"char": "a", "start": 0.7}],
        }]}
        data = whisper.convert_to_transcribed_data(result_aligned)
        self.assertEqual([c["char"] for c in data[0].char_timings], ["h", "o", "l", "a"])
        self.assertEqual(data[0].char_timings[2]["start"], 0.5)


if __name__ == "__main__":
    unittest.main()
