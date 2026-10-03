"""Tests for midi_creator.py"""

import unittest

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Midi.midi_creator import create_midi_instrument


class CreateMidiInstrumentTest(unittest.TestCase):
    def test_creates_one_note_per_segment(self):
        segments = [MidiSegment("C4", 0.0, 1.0, "la"), MidiSegment("D4", 1.0, 2.0, "la")]

        instrument = create_midi_instrument(segments)

        self.assertEqual(len(instrument.notes), 2)

    def test_skips_segments_without_duration(self):
        segments = [
            MidiSegment("C4", 0.0, 1.0, "la"),
            MidiSegment("D4", 2.0, 2.0, "zero length"),
            MidiSegment("E4", 3.0, 2.5, "negative length"),
        ]

        instrument = create_midi_instrument(segments)

        self.assertEqual(len(instrument.notes), 1)


if __name__ == "__main__":
    unittest.main()
