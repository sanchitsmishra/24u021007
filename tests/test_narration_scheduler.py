"""Speech-gap scheduling for Piper narration clips."""

from __future__ import annotations

import unittest

from app.narration_scheduler import (
    DEFAULT_SPEECH_SAFETY_MARGIN_SECONDS,
    expand_speech_intervals,
    schedule_narration_clips,
)
from app.narration_tts import NarrationClip
from app.speech_detector import SpeechTimeline, TimeInterval


def clip(timestamp: float, duration: float | None, description: str = "A person walks.") -> NarrationClip:
    return NarrationClip(
        timestamp_seconds=timestamp,
        description=description,
        clip_id="a" * 32 if duration else None,
        duration_seconds=duration,
    )


class SpeechGapSchedulingTests(unittest.TestCase):
    def test_clip_fits_at_preferred_time_in_a_silent_video(self) -> None:
        scheduled = schedule_narration_clips(
            [clip(1.0, 2.0)],
            SpeechTimeline(duration_seconds=10.0, speech_intervals=()),
        )
        self.assertEqual(scheduled[0].scheduled_start_seconds, 1.0)
        self.assertEqual(scheduled[0].duration_seconds, 2.0)
        self.assertEqual(scheduled[0].timestamp_seconds, 1.0)

    def test_clip_that_cannot_finish_before_speech_is_skipped(self) -> None:
        timeline = SpeechTimeline(
            duration_seconds=10.0,
            speech_intervals=(TimeInterval(2.0, 9.0),),
        )
        scheduled = schedule_narration_clips(
            [clip(1.0, 5.0)],
            timeline,
            safety_margin_seconds=0.25,
        )
        self.assertIsNone(scheduled[0].scheduled_start_seconds)
        self.assertEqual(scheduled[0].description, "A person walks.")

    def test_clip_during_speech_moves_to_the_next_fitting_gap(self) -> None:
        timeline = SpeechTimeline(
            duration_seconds=8.0,
            speech_intervals=(TimeInterval(0.5, 3.0),),
        )
        scheduled = schedule_narration_clips(
            [clip(1.0, 1.0)],
            timeline,
            safety_margin_seconds=0.25,
        )
        self.assertEqual(scheduled[0].scheduled_start_seconds, 3.25)

    def test_clip_near_speech_boundary_is_blocked_by_safety_margin(self) -> None:
        timeline = SpeechTimeline(
            duration_seconds=10.0,
            speech_intervals=(TimeInterval(5.0, 7.0),),
        )
        too_close = schedule_narration_clips(
            [clip(4.0, 0.9)],
            timeline,
            safety_margin_seconds=0.25,
        )
        self.assertNotEqual(too_close[0].scheduled_start_seconds, 4.0)
        self.assertEqual(too_close[0].scheduled_start_seconds, 7.25)

        fits = schedule_narration_clips(
            [clip(4.0, 0.6)],
            timeline,
            safety_margin_seconds=0.25,
        )
        self.assertEqual(fits[0].scheduled_start_seconds, 4.0)

    def test_safety_margin_shrinks_usable_gaps(self) -> None:
        speech = (TimeInterval(5.0, 6.0),)
        padded = expand_speech_intervals(speech, 0.25, 10.0)
        self.assertEqual(padded, (TimeInterval(4.75, 6.25),))
        self.assertEqual(DEFAULT_SPEECH_SAFETY_MARGIN_SECONDS, 0.25)

    def test_later_clip_does_not_overlap_an_earlier_scheduled_clip(self) -> None:
        timeline = SpeechTimeline(duration_seconds=12.0, speech_intervals=())
        scheduled = schedule_narration_clips(
            [clip(1.0, 3.0, "First scene."), clip(2.0, 1.5, "Second scene.")],
            timeline,
        )
        self.assertEqual(scheduled[0].scheduled_start_seconds, 1.0)
        self.assertEqual(scheduled[1].scheduled_start_seconds, 4.0)

    def test_clip_without_duration_is_not_scheduled(self) -> None:
        scheduled = schedule_narration_clips(
            [clip(1.0, None)],
            SpeechTimeline(duration_seconds=5.0, speech_intervals=()),
        )
        self.assertIsNone(scheduled[0].scheduled_start_seconds)
        self.assertIsNone(scheduled[0].clip_id)

    def test_clip_that_fits_only_after_another_clip_still_must_beat_speech(self) -> None:
        timeline = SpeechTimeline(
            duration_seconds=10.0,
            speech_intervals=(TimeInterval(5.0, 8.0),),
        )
        scheduled = schedule_narration_clips(
            [clip(0.5, 3.0), clip(1.0, 2.5)],
            timeline,
            safety_margin_seconds=0.25,
        )
        self.assertEqual(scheduled[0].scheduled_start_seconds, 0.5)
        # 0.5 + 3.0 = 3.5, then 2.5s more would run to 6.0 and overlap padded speech at 4.75.
        self.assertIsNone(scheduled[1].scheduled_start_seconds)


if __name__ == "__main__":
    unittest.main()
