"""Plan how a long vocal track is cut into chunks, cutting only inside silences"""


def plan_chunks(total_seconds: float,
                silences: list[tuple[float, float]],
                max_chunk_seconds: float = 60.0,
                min_chunk_seconds: float = 15.0,
                min_cut_silence_seconds: float = 0.3) -> list[tuple[float, float]]:
    """Return consecutive (start, end) chunks covering [0, total_seconds].

    Every chunk is at most max_chunk_seconds long. The cut is placed in the middle of the
    last silence (at least min_cut_silence_seconds long) that fits in the window, so no word
    is split and no overlap is needed. If a window contains no usable silence it is cut hard.
    """
    if total_seconds <= 0:
        return []
    if max_chunk_seconds <= 0 or total_seconds <= max_chunk_seconds:
        return [(0.0, total_seconds)]

    cut_points = sorted(
        (s + e) / 2 for s, e in silences if e - s >= min_cut_silence_seconds
    )

    chunks = []
    start = 0.0
    while total_seconds - start > max_chunk_seconds:
        window_end = start + max_chunk_seconds
        candidates = [c for c in cut_points if start + min_chunk_seconds <= c <= window_end]
        end = candidates[-1] if candidates else window_end
        chunks.append((start, end))
        start = end
    chunks.append((start, total_seconds))
    return chunks
