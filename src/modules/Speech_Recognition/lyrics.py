"""Use known lyrics instead of the (error-prone) speech recognition text of whisper"""
import re
from bisect import bisect_right
from difflib import SequenceMatcher

# Lyric words before the first / after the last recognised word that still count as part of
# the first / last segment (whisper often misses the very first or last words of a song)
_EDGE_TOLERANCE_WORDS = 3

_SECTION_HEADER = re.compile(r"^\s*[\[(].*[\])]\s*$")


def normalize_token(token: str) -> str:
    """Lower-case letters and digits only, used to compare recognised words with lyrics"""
    return "".join(c for c in token.lower() if c.isalnum())


def parse_lyrics(text: str) -> list[str]:
    """Words of the lyrics, without blank lines and [Chorus]-like section headers"""
    words = []
    for line in text.splitlines():
        if not line.strip() or _SECTION_HEADER.match(line):
            continue
        words.extend(line.split())
    return words


def load_lyrics(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8-sig") as file:
        return parse_lyrics(file.read())


def map_lyrics_to_segments(asr_segment_words: list[list[str]],
                           lyric_words: list[str]) -> tuple[list[list[str]], float]:
    """Assign the lyric words to the recognised segments.

    The recognised words are matched against the lyrics to find anchor points. Between two
    anchors the lyrics are spread linearly, outside of them one to one (up to a few words at
    the very start and end of the song are kept). Returns the lyric
    words per segment and the share of recognised words that matched the lyrics exactly.
    """
    asr_norm = [normalize_token(w) for words in asr_segment_words for w in words]
    lyric_norm = [normalize_token(w) for w in lyric_words]

    blocks = [b for b in SequenceMatcher(None, asr_norm, lyric_norm, autojunk=False).get_matching_blocks()
              if b.size > 0]
    matched = sum(b.size for b in blocks)
    match_ratio = matched / len(asr_norm) if asr_norm else 0.0

    if not blocks:
        # Nothing to anchor on: spread the lyrics evenly over the segments
        anchors = [(0, 0), (len(asr_norm), len(lyric_words))]
    else:
        anchors = []
        for block in blocks:
            anchors.append((block.a, block.b))
            anchors.append((block.a + block.size, block.b + block.size))

    asr_anchor_positions = [a for a, _ in anchors]

    def to_lyric_index(asr_index: int) -> int:
        if not blocks:
            return round(asr_index * len(lyric_words) / max(len(asr_norm), 1))
        if asr_index <= anchors[0][0]:
            return anchors[0][1] - (anchors[0][0] - asr_index)
        if asr_index >= anchors[-1][0]:
            return anchors[-1][1] + (asr_index - anchors[-1][0])
        right = bisect_right(asr_anchor_positions, asr_index)
        (a0, l0), (a1, l1) = anchors[right - 1], anchors[right]
        if a1 == a0:
            return l0
        return round(l0 + (l1 - l0) * (asr_index - a0) / (a1 - a0))

    result = []
    cursor = 0
    boundary = 0
    last_index = len(asr_segment_words) - 1
    for index, words in enumerate(asr_segment_words):
        boundary += len(words)
        low = max(cursor, to_lyric_index(boundary - len(words)), 0)
        high = min(max(low, to_lyric_index(boundary)), len(lyric_words))
        if index == 0 and low <= _EDGE_TOLERANCE_WORDS:
            low = 0
        if index == last_index and len(lyric_words) - high <= _EDGE_TOLERANCE_WORDS:
            high = len(lyric_words)
        result.append(lyric_words[low:high])
        cursor = high
    return result, match_ratio
