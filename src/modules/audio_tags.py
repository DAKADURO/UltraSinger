"""Read metadata embedded in audio files (ID3, Vorbis, MP4 ...)"""

import re
from dataclasses import dataclass
from typing import Optional

import mutagen


@dataclass
class AudioTags:
    artist: Optional[str] = None
    title: Optional[str] = None
    year: Optional[str] = None


def __first(tags, key: str) -> Optional[str]:
    values = tags.get(key)
    if not values:
        return None
    value = str(values[0]).strip()
    return value or None


def read_audio_tags(file_path: str) -> AudioTags:
    """Read artist, title and year from the file tags. Missing or unreadable tags give empty values."""
    try:
        tags = mutagen.File(file_path, easy=True)
    except Exception:
        return AudioTags()
    if tags is None:
        return AudioTags()

    year = None
    date = __first(tags, "originaldate") or __first(tags, "date")
    if date:
        match = re.match(r"\d{4}", date)
        year = match.group(0) if match else None

    return AudioTags(artist=__first(tags, "artist"), title=__first(tags, "title"), year=year)
