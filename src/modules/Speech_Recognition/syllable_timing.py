"""Derive syllable start/end times from the character timings of the forced alignment

WhisperX can return the time of every character. Splitting a word into syllables with the
character timings keeps held vowels long and consonants short, instead of giving every
syllable the same duration.
"""
import copy

from modules.Speech_Recognition.TranscribedData import TranscribedData

MIN_SYLLABLE_SECONDS = 0.04


def _visible(text: str) -> str:
    """The text without whitespace"""
    return "".join(text.split())


def assign_chars_to_words(words: list[str], chars: list[dict]) -> list[list[dict]] | None:
    """Split the flat character list of a segment into one list per word.

    Returns None if the characters do not match the words, so the caller can fall back.
    """
    stream = [c for c in chars if c.get("char") is not None and not str(c["char"]).isspace()]
    result = []
    position = 0
    for word in words:
        expected = _visible(word)
        word_chars = stream[position:position + len(expected)]
        if len(word_chars) != len(expected):
            return None
        for expected_char, char in zip(expected, word_chars):
            if str(char["char"]).lower() != expected_char.lower():
                return None
        result.append(word_chars)
        position += len(expected)
    return result


def _fill_missing_starts(starts: list[float | None], word_start: float, word_end: float) -> list[float]:
    """Interpolate characters without timing (digits, symbols) between their timed neighbours"""
    known = [i for i, s in enumerate(starts) if s is not None]
    if not known:
        count = len(starts)
        return [word_start + (word_end - word_start) * i / count for i in range(count)]

    filled = list(starts)
    for i in range(len(filled)):
        if filled[i] is not None:
            continue
        previous = max((k for k in known if k < i), default=None)
        following = min((k for k in known if k > i), default=None)
        if previous is None:
            filled[i] = word_start
        elif following is None:
            filled[i] = starts[previous]
        else:
            ratio = (i - previous) / (following - previous)
            filled[i] = starts[previous] + (starts[following] - starts[previous]) * ratio
    return filled


def syllable_times_from_chars(word_start: float, word_end: float, syllables: list[str],
                              word_chars: list[dict]) -> list[tuple[float, float]] | None:
    """Return (start, end) for every syllable, contiguous and covering the whole word.

    A syllable starts when its first character starts and lasts until the next syllable
    starts. None is returned when the data is inconsistent or a syllable would be shorter
    than MIN_SYLLABLE_SECONDS, so the caller can use an even split instead.
    """
    if len(syllables) < 2 or word_end <= word_start:
        return None
    if sum(len(_visible(s)) for s in syllables) != len(word_chars):
        return None

    starts = _fill_missing_starts([c.get("start") for c in word_chars], word_start, word_end)

    boundaries = [word_start]
    index = 0
    for syllable in syllables[:-1]:
        index += len(_visible(syllable))
        boundaries.append(starts[index])
    boundaries.append(word_end)

    boundaries = [min(max(b, word_start), word_end) for b in boundaries]
    for first, second in zip(boundaries, boundaries[1:]):
        if second - first < MIN_SYLLABLE_SECONDS:
            return None
    return list(zip(boundaries[:-1], boundaries[1:]))


def add_hyphen_to_data(transcribed_data: list[TranscribedData],
                       hyphen_words: list[list[str] | None]) -> list[TranscribedData]:
    """Split every hyphenated word into one entry per syllable and return the new data list.

    The syllable times come from the character alignment when available, otherwise the word
    duration is divided evenly.
    """
    new_data = []

    for data, syllables in zip(transcribed_data, hyphen_words):
        if not syllables:
            new_data.append(data)
            continue

        times = None
        if data.char_timings:
            times = syllable_times_from_chars(data.start, data.end, syllables, data.char_timings)
        if times is None:
            chunk_duration = (data.end - data.start) / len(syllables)
            times = [(data.start + chunk_duration * i, data.start + chunk_duration * (i + 1))
                     for i in range(len(syllables))]
            times[-1] = (times[-1][0], data.end)

        for index, (syllable, (start, end)) in enumerate(zip(syllables, times)):
            dup = copy.copy(data)
            dup.char_timings = None
            dup.start = start
            dup.end = end
            dup.word = syllable
            dup.is_hyphen = True
            dup.is_word_end = index == len(syllables) - 1
            new_data.append(dup)

    return new_data
