"""Tests for chunk_planner.py"""

import unittest

from src.modules.Audio.chunk_planner import plan_chunks


class PlanChunksTest(unittest.TestCase):
    def test_short_audio_is_a_single_chunk(self):
        self.assertEqual(plan_chunks(40.0, [(10.0, 11.0)], max_chunk_seconds=60), [(0.0, 40.0)])

    def test_empty_audio_has_no_chunks(self):
        self.assertEqual(plan_chunks(0.0, []), [])

    def test_cuts_in_the_middle_of_the_last_silence_in_window(self):
        silences = [(20.0, 21.0), (50.0, 52.0), (70.0, 72.0)]
        chunks = plan_chunks(100.0, silences, max_chunk_seconds=60, min_chunk_seconds=15)
        self.assertEqual(chunks, [(0.0, 51.0), (51.0, 100.0)])

    def test_chunks_are_contiguous_and_bounded(self):
        silences = [(i * 25.0, i * 25.0 + 0.5) for i in range(1, 20)]
        chunks = plan_chunks(480.0, silences, max_chunk_seconds=60)
        self.assertEqual(chunks[0][0], 0.0)
        self.assertEqual(chunks[-1][1], 480.0)
        for (_, end), (start, _) in zip(chunks, chunks[1:]):
            self.assertEqual(end, start)
        self.assertTrue(all(end - start <= 60 for start, end in chunks))

    def test_hard_cut_when_no_silence_available(self):
        chunks = plan_chunks(130.0, [], max_chunk_seconds=60)
        self.assertEqual(chunks, [(0.0, 60.0), (60.0, 120.0), (120.0, 130.0)])

    def test_ignores_silences_too_short_to_cut(self):
        chunks = plan_chunks(100.0, [(40.0, 40.1)], max_chunk_seconds=60)
        self.assertEqual(chunks[0], (0.0, 60.0))


if __name__ == "__main__":
    unittest.main()
