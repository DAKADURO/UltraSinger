"""Synced lyrics from LRCLIB (https://lrclib.net) or from a local .lrc file"""

import json
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

from Levenshtein import ratio

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted

LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
USER_AGENT = "UltraSinger (https://github.com/rakuri255/UltraSinger)"
MAX_RETRIES = 3
# A lyrics version whose duration differs more than this from the audio is another edition (live, remix ...)
MAX_DURATION_DIFFERENCE = 10.0
# Maximum distance between the time of a transcribed segment and a lyric line to consider them the same line
MATCH_WINDOW = 15.0
MIN_MATCH_RATIO = 0.6
MAX_LINES_PER_SEGMENT = 4
MIN_OFFSET_TO_TRY = 0.5  # smaller estimated shifts are not worth an extra alignment
MIN_SCORE_GAIN = 0.02  # a shifted alignment must be clearly better than the unshifted one
# Voice detection used to find where the last word of a line ends
VOICE_FRAME_SECONDS = 0.02
VOICE_THRESHOLD = 0.05  # fraction of the loud (95th percentile) level of the audio
VOICE_END_MARGIN = 0.05
MAX_PAUSE_SECONDS = 0.5  # a longer break in the voice is not part of the word
MIN_WORD_SECONDS = 0.05

LRC_LINE = re.compile(r"^((?:\[\d+:\d+(?:[.:]\d+)?\])+)(.*)$")
LRC_TIME = re.compile(r"\[(\d+):(\d+)(?:[.:](\d+))?\]")


@dataclass
class LyricLine:
    start: float
    text: str


def parse_lrc(lrc_text: str) -> list[LyricLine]:
    """Parse the lines of a synced lyrics text ([mm:ss.xx] words). Lines without words are dropped."""
    lines = []
    for raw_line in lrc_text.splitlines():
        match = LRC_LINE.match(raw_line.strip())
        if not match:
            continue
        text = match.group(2).strip()
        if not text:
            continue
        for minutes, seconds, fraction in LRC_TIME.findall(match.group(1)):
            fraction_seconds = int(fraction) / 10 ** len(fraction) if fraction else 0.0
            lines.append(LyricLine(int(minutes) * 60 + int(seconds) + fraction_seconds, text))
    return sorted(lines, key=lambda line: line.start)


def read_lrc_file(file_path: str) -> list[LyricLine]:
    with open(file_path, encoding="utf-8-sig") as file:
        return parse_lrc(file.read())


def pick_best_result(results: list[dict], audio_duration: Optional[float]) -> Optional[dict]:
    """The result with synced lyrics whose duration is closest to the audio duration"""
    candidates = [result for result in results if result.get("syncedLyrics")]
    if not candidates:
        return None
    if audio_duration is None:
        return candidates[0]

    best = min(candidates, key=lambda result: abs((result.get("duration") or 0) - audio_duration))
    if abs((best.get("duration") or 0) - audio_duration) > MAX_DURATION_DIFFERENCE:
        return None
    return best


def __request(params: dict) -> Optional[list[dict]]:
    url = f"{LRCLIB_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for _ in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(1)
    return None


def fetch_synced_lyrics(artist: str, title: str, audio_duration: Optional[float]) -> Optional[list[LyricLine]]:
    """Search synced lyrics on LRCLIB. Sends only artist and title to lrclib.net."""
    print(f"{ULTRASINGER_HEAD} Searching lyrics on {blue_highlighted('LRCLIB')} for {blue_highlighted(f'{artist} - {title}')}")
    results = __request({"artist_name": artist, "track_name": title})
    if not results:
        results = __request({"q": f"{artist} {title}"})

    best = pick_best_result(results or [], audio_duration)
    if best is None:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('No matching synced lyrics found')} - using the transcription")
        return None

    lines = parse_lrc(best["syncedLyrics"])
    print(f"{ULTRASINGER_HEAD} Found {blue_highlighted(str(len(lines)))} lyric lines ({best.get('albumName') or 'unknown album'})")
    return lines or None


def __normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def estimate_offset(segments: list[dict], lines: list[LyricLine]) -> float:
    """Time shift (seconds) between the lyrics and the audio.

    A transcribed segment often contains several lyric lines, so it is compared with groups of up to
    MAX_LINES_PER_SEGMENT consecutive lines near it. The start of the best matching group is the
    start of the segment in the lyrics. The median of these differences is used, so a few wrong
    matches do not matter."""
    differences = []
    for segment in segments:
        text = __normalize(segment.get("text", ""))
        if not text:
            continue
        best_ratio, best_start = 0.0, None
        for index, line in enumerate(lines):
            if abs(line.start - segment["start"]) > MATCH_WINDOW:
                continue
            for count in range(1, MAX_LINES_PER_SEGMENT + 1):
                group = lines[index:index + count]
                group_ratio = ratio(text, __normalize(" ".join(member.text for member in group)))
                if group_ratio > best_ratio:
                    best_ratio, best_start = group_ratio, line.start
        if best_start is not None and best_ratio >= MIN_MATCH_RATIO:
            differences.append(segment["start"] - best_start)
    return statistics.median(differences) if differences else 0.0


def candidate_offsets(estimated: float) -> list[float]:
    """Time shifts to try: none (lyrics usually match the audio) and the estimated one"""
    return [0.0] if abs(estimated) < MIN_OFFSET_TO_TRY else [0.0, estimated]


def alignment_score(result_aligned: dict) -> float:
    """Mean confidence of the aligned words. Words in the wrong place of the audio get a low confidence."""
    scores = [
        word["score"]
        for segment in result_aligned.get("segments", [])
        for word in segment.get("words", [])
        if "score" in word
    ]
    return sum(scores) / len(scores) if scores else 0.0


def trim_segment_ends(segments: list[dict], audio, sample_rate: int = 16000) -> None:
    """Shorten the last word of every aligned segment to where the voice stops.

    The lyric windows reach to the next line, so the forced alignment stretches the last word of a line
    over the pause that follows it. Only the energy of the (vocal) audio is used to find the real end."""
    import numpy as np

    hop = int(sample_rate * VOICE_FRAME_SECONDS)
    frame_count = len(audio) // hop
    if frame_count == 0:
        return
    frames = np.asarray(audio[:frame_count * hop], dtype=np.float64).reshape(frame_count, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    voiced = rms > VOICE_THRESHOLD * np.percentile(rms, 95)

    for segment in segments:
        words = [word for word in segment.get("words", []) if "start" in word and "end" in word]
        if not words:
            continue
        last = words[-1]
        first_frame = int(last["start"] / VOICE_FRAME_SECONDS)
        end_frame = min(int(last["end"] / VOICE_FRAME_SECONDS) + 1, frame_count)
        voiced_frames = np.nonzero(voiced[first_frame:end_frame])[0]
        if len(voiced_frames) == 0:
            continue
        # The word ends where the voice breaks off; what sounds after a long pause is the next line
        pauses = np.nonzero(np.diff(voiced_frames) > MAX_PAUSE_SECONDS / VOICE_FRAME_SECONDS)[0]
        last_voiced_frame = voiced_frames[pauses[0]] if len(pauses) else voiced_frames[-1]
        voice_end = (first_frame + last_voiced_frame + 1) * VOICE_FRAME_SECONDS + VOICE_END_MARGIN
        if last["start"] + MIN_WORD_SECONDS < voice_end < last["end"]:
            last["end"] = voice_end


def lines_to_segments(lines: list[LyricLine], offset: float, audio_duration: Optional[float]) -> list[dict]:
    """Segments (text with a time window) for the forced alignment of the lyrics"""
    segments = []
    for index, line in enumerate(lines):
        start = max(0.0, line.start + offset - 0.3)
        if index + 1 < len(lines):
            end = lines[index + 1].start + offset
        else:
            end = line.start + offset + 8.0
        if audio_duration is not None:
            end = min(end, audio_duration)
        if end - start < 0.5:
            continue
        segments.append({"text": line.text, "start": start, "end": end})
    return segments
