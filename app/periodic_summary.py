"""Fixed 10-second visual summaries using the existing captioning models.

This path ignores speech. Clips are meant to pause the video at each window
end, including a final short remainder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.caption_service import NarrationMoment, VideoCaptionService

PERIODIC_INTERVAL_SECONDS = 10.0
MAX_FRAMES_PER_WINDOW = 3
NARRATION_SIMILARITY_THRESHOLD = 0.62


@dataclass(frozen=True)
class PeriodicSegment:
    segment_start_seconds: float
    segment_end_seconds: float
    description: str


def segment_windows(
    duration_seconds: float, interval_seconds: float = PERIODIC_INTERVAL_SECONDS
) -> tuple[tuple[float, float], ...]:
    """Split a duration into [0, 10], [10, 20], … plus a final short remainder."""
    if duration_seconds <= 0 or interval_seconds <= 0:
        return ()
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_seconds - 1e-9:
        end = min(start + interval_seconds, duration_seconds)
        windows.append((round(start, 3), round(end, 3)))
        start = end
    return tuple(windows)


def frame_timestamps_for_window(
    start_seconds: float, end_seconds: float, max_frames: int = MAX_FRAMES_PER_WINDOW
) -> tuple[float, ...]:
    """Pick 1–3 timestamps inside a window, clustered away from the exact edges."""
    duration = end_seconds - start_seconds
    if duration <= 0 or max_frames < 1:
        return ()
    if duration < 2.0 or max_frames == 1:
        fractions = (0.5,)
    elif duration < 6.0 or max_frames == 2:
        fractions = (0.3, 0.7)
    else:
        fractions = (0.15, 0.5, 0.85)
    stamps = [round(start_seconds + duration * fraction, 3) for fraction in fractions[:max_frames]]
    unique: list[float] = []
    for stamp in stamps:
        if stamp < start_seconds:
            stamp = start_seconds
        if stamp > end_seconds:
            stamp = end_seconds
        if unique and abs(stamp - unique[-1]) < 0.05:
            continue
        unique.append(stamp)
    return tuple(unique)


def summarise_window_captions(captions: list[str]) -> str:
    """Combine distinct BLIP captions into one chronological sentence."""

    cleaned = [
        caption.strip()
        for caption in captions
        if caption and caption.strip()
    ]

    if not cleaned:
        return "The scene continues."

    unique: list[str] = []

    for caption in cleaned:
        repeats = any(
            VideoCaptionService._caption_similarity(caption, previous)
            >= NARRATION_SIMILARITY_THRESHOLD
            for previous in unique
        )

        if not repeats:
            unique.append(caption)

    if len(unique) == 1:
        return VideoCaptionService._speakable_sentence(unique[0])

    observations = [
        caption.rstrip(" .!?")
        for caption in unique
    ]

    combined = (
        "The sequence shows "
        + ", then ".join(observations)
        + "."
    )

    return VideoCaptionService._speakable_sentence(combined)



def build_periodic_segments(
    service: VideoCaptionService, video_path: Path
) -> list[PeriodicSegment]:
    """Caption 1–3 stills per 10-second window with the existing local models."""
    duration_seconds = service.video_duration_seconds(video_path)
    segments: list[PeriodicSegment] = []
    for start_seconds, end_seconds in segment_windows(duration_seconds):
        captions: list[str] = []
        for timestamp in frame_timestamps_for_window(start_seconds, end_seconds):
            sampled = service.read_frame_at(video_path, timestamp)
            if sampled is None:
                continue
            caption = service.caption_still(sampled.image)
            if caption.strip():
                captions.append(caption)
        if not captions:
            continue
        segments.append(
            PeriodicSegment(
                segment_start_seconds=start_seconds,
                segment_end_seconds=end_seconds,
                description=summarise_window_captions(captions),
            )
        )
    return segments


def segments_to_moments(segments: list[PeriodicSegment]) -> list[NarrationMoment]:
    """Piper synthesizes from NarrationMoment text; pause time is the window end."""
    return [
        NarrationMoment(
            timestamp_seconds=segment.segment_end_seconds,
            description=segment.description,
        )
        for segment in segments
    ]
