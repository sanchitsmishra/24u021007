"""Place Piper clips in speech-free gaps using measured clip durations.

Visual captioning still decides *what* to say. This module only decides *when*
a clip may start so it finishes before the next padded speech interval.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.narration_tts import NarrationClip
from app.speech_detector import SpeechTimeline, TimeInterval, merge_speech_intervals, speech_free_intervals

# Keep a little silence on either side of detected dialogue so narration does
# not start or end on a word boundary.
DEFAULT_SPEECH_SAFETY_MARGIN_SECONDS = 0.25


@dataclass(frozen=True)
class ScheduledNarration:
    timestamp_seconds: float
    description: str
    clip_id: str | None
    duration_seconds: float | None
    scheduled_start_seconds: float | None


def expand_speech_intervals(
    intervals: tuple[TimeInterval, ...] | list[TimeInterval],
    safety_margin_seconds: float,
    duration_seconds: float,
) -> tuple[TimeInterval, ...]:
    """Widen speech windows by a safety margin, then merge overlaps."""
    if safety_margin_seconds < 0:
        raise ValueError("Speech safety margin cannot be negative.")
    padded = [
        TimeInterval(
            start_seconds=max(0.0, interval.start_seconds - safety_margin_seconds),
            end_seconds=min(duration_seconds, interval.end_seconds + safety_margin_seconds),
        )
        for interval in intervals
        if interval.end_seconds > interval.start_seconds
    ]
    return merge_speech_intervals(padded, min_silence_gap_seconds=0.0)


def _media_duration_seconds(clips: list[NarrationClip], timeline: SpeechTimeline) -> float:
    duration = timeline.duration_seconds
    if duration > 0:
        return duration
    inferred = 0.0
    for clip in clips:
        clip_end = clip.timestamp_seconds + (clip.duration_seconds or 0.0)
        inferred = max(inferred, clip_end)
    return inferred


def schedule_narration_clips(
    clips: list[NarrationClip],
    timeline: SpeechTimeline,
    safety_margin_seconds: float = DEFAULT_SPEECH_SAFETY_MARGIN_SECONDS,
) -> list[ScheduledNarration]:
    """Assign a start time only when the full clip fits in a speech-free gap.

    Preferred start is the original visual timestamp. If that window is blocked,
    the next later gap is tried. A clip that cannot finish before speech is
    skipped instead of overlapping dialogue. Earlier clips occupy space in a
    gap so later clips cannot overlap them either.
    """
    duration_seconds = _media_duration_seconds(clips, timeline)
    blocked = expand_speech_intervals(
        timeline.speech_intervals, safety_margin_seconds, duration_seconds
    )
    gaps = speech_free_intervals(duration_seconds, blocked)
    gap_cursors = [gap.start_seconds for gap in gaps]

    scheduled: list[ScheduledNarration] = []
    for clip in clips:
        placement = _first_fitting_placement(
            preferred_start=clip.timestamp_seconds,
            clip_duration=clip.duration_seconds,
            gaps=gaps,
            gap_cursors=gap_cursors,
        )
        start = None
        if placement is not None:
            start, gap_index = placement
            gap_cursors[gap_index] = start + (clip.duration_seconds or 0.0)
        scheduled.append(
            ScheduledNarration(
                timestamp_seconds=clip.timestamp_seconds,
                description=clip.description,
                clip_id=clip.clip_id,
                duration_seconds=clip.duration_seconds,
                scheduled_start_seconds=start,
            )
        )
    return scheduled


def _first_fitting_placement(
    preferred_start: float,
    clip_duration: float | None,
    gaps: tuple[TimeInterval, ...],
    gap_cursors: list[float],
) -> tuple[float, int] | None:
    if clip_duration is None or clip_duration <= 0:
        return None

    for index, gap in enumerate(gaps):
        if preferred_start > gap.end_seconds:
            continue
        earliest = max(gap.start_seconds, gap_cursors[index], preferred_start)
        if earliest + clip_duration <= gap.end_seconds + 1e-9:
            return round(earliest, 3), index
    return None
