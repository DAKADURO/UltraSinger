"""Turn the frequencies of a syllable into a clean note.

All pitches here are MIDI numbers as floats (69 = A4 = 440 Hz)."""

from typing import Optional

import numpy as np

# Frequencies outside of this range are detection errors (consonants, noise), not a singing voice
MIN_SINGING_HZ = 70.0
MAX_SINGING_HZ = 1100.0
# A note further away than this from the notes around it is checked for an octave error
OCTAVE_CHECK_SEMITONES = 7
OCTAVE_NEIGHBOURS = 6
# Notes between two semitones are only moved into the key if they are further away than this from a semitone
CLEAR_NOTE_SEMITONES = 0.25
# The key is only used when the sung notes fit it clearly
MIN_KEY_CORRELATION = 0.5

# Krumhansl-Kessler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def hz_to_midi(hz) -> np.ndarray:
    return 69 + 12 * np.log2(np.asarray(hz, dtype=float) / 440.0)


def steady_pitch(frequencies: list[float], confidences: list[float], confidence_threshold: float = 0.4,
                 window: float = 0.5) -> Optional[float]:
    """The steadiest pitch of a syllable: the mean of the frames inside the 0.5 semitone window that holds
    the most frames. Frames outside the singing range or with low confidence are ignored.
    Returns None if no frame is usable."""
    values = [
        hz for hz, confidence in zip(frequencies, confidences)
        if confidence > confidence_threshold and MIN_SINGING_HZ <= hz <= MAX_SINGING_HZ
    ]
    if not values:
        return None
    midi = np.sort(hz_to_midi(values))
    best_start, best_count, low = 0, 0, 0
    for high in range(len(midi)):
        while midi[high] - midi[low] > window:
            low += 1
        if high - low + 1 > best_count:
            best_start, best_count = low, high - low + 1
    return float(np.mean(midi[best_start:best_start + best_count]))


def fill_missing(pitches: list[Optional[float]], default: float = 60.0) -> list[float]:
    """Notes without usable frames get the pitch of the closest note before (or else after) them"""
    result = list(pitches)
    last = None
    for index, pitch in enumerate(result):
        if pitch is None:
            result[index] = last
        else:
            last = pitch
    following = None
    for index in range(len(result) - 1, -1, -1):
        if result[index] is None:
            result[index] = following
        else:
            following = result[index]
    return [default if pitch is None else pitch for pitch in result]


def fix_octaves(pitches: list[float]) -> list[float]:
    """Move notes that are far away from the notes around them by whole octaves towards them.

    Pitch detectors jump an octave (or more) for single syllables; real melodies rarely do."""
    result = list(pitches)
    for index, pitch in enumerate(pitches):
        low, high = max(0, index - OCTAVE_NEIGHBOURS), min(len(pitches), index + OCTAVE_NEIGHBOURS + 1)
        neighbours = [p for position, p in enumerate(pitches[low:high], low) if position != index]
        if len(neighbours) < 3:
            continue
        median = float(np.median(neighbours))
        best = min((pitch + 12 * shift for shift in (-2, -1, 0, 1, 2)), key=lambda candidate: abs(candidate - median))
        if abs(pitch - median) > OCTAVE_CHECK_SEMITONES and abs(best - median) < abs(pitch - median):
            result[index] = best
    return result


def detect_key_from_pitches(pitches: list[float], durations: list[float]) -> Optional[tuple[str, str, float]]:
    """(root, mode, correlation) of the key that fits the sung notes best. None if there are no notes."""
    histogram = np.zeros(12)
    for pitch, duration in zip(pitches, durations):
        histogram[int(round(pitch)) % 12] += duration
    if histogram.sum() == 0:
        return None
    best = None
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            correlation = float(np.corrcoef(histogram, np.roll(profile, tonic))[0, 1])
            if best is None or correlation > best[2]:
                best = (NOTE_NAMES[tonic], mode, correlation)
    return best


def allowed_pitch_classes(root: str, mode: str) -> set[int]:
    scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    return {(NOTE_NAMES.index(root) + interval) % 12 for interval in scale}


def soft_quantize(pitch: float, allowed: set[int]) -> int:
    """Round to a semitone. A note between two semitones that is not in the key is moved into the key;
    a note that clearly sits on a semitone outside the key is a real chromatic note and stays."""
    rounded = int(round(pitch))
    if rounded % 12 in allowed or abs(pitch - rounded) < CLEAR_NOTE_SEMITONES:
        return rounded
    candidates = [semitone for semitone in (int(np.floor(pitch)), int(np.ceil(pitch))) if semitone % 12 in allowed]
    return min(candidates, key=lambda semitone: abs(semitone - pitch)) if candidates else rounded
