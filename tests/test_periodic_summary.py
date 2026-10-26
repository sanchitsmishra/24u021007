"""Segment boundaries and mode separation for periodic visual summaries."""

from __future__ import annotations

from pathlib import Path
import re
import unittest

from app.narration_scheduler import schedule_narration_clips
from app.narration_tts import NarrationClip
from app.periodic_summary import (
    PeriodicSegment,
    frame_timestamps_for_window,
    segment_windows,
    segments_to_moments,
    summarise_window_captions,
)
from app.schemas import AudioDescription, CaptionResponse, PeriodicSummary
from app.speech_detector import SpeechTimeline, TimeInterval


class SegmentBoundaryTests(unittest.TestCase):
    def test_ten_second_windows_and_final_short_segment(self) -> None:
        self.assertEqual(segment_windows(25.0), ((0.0, 10.0), (10.0, 20.0), (20.0, 25.0)))
        self.assertEqual(segment_windows(20.0), ((0.0, 10.0), (10.0, 20.0)))
        self.assertEqual(segment_windows(8.0), ((0.0, 8.0),))
        self.assertEqual(segment_windows(10.0), ((0.0, 10.0),))
        self.assertEqual(segment_windows(0.0), ())

    def test_window_uses_one_to_three_frame_timestamps(self) -> None:
        self.assertEqual(len(frame_timestamps_for_window(0.0, 1.5)), 1)
        self.assertEqual(len(frame_timestamps_for_window(0.0, 5.0)), 2)
        self.assertEqual(len(frame_timestamps_for_window(0.0, 10.0)), 3)
        for stamp in frame_timestamps_for_window(10.0, 20.0):
            self.assertGreaterEqual(stamp, 10.0)
            self.assertLessEqual(stamp, 20.0)

    def test_summarise_window_keeps_a_single_speakable_sentence(self) -> None:
        sentence = summarise_window_captions(["a man standing", "a man standing"])
        self.assertTrue(sentence.endswith("."))
        self.assertIn("man", sentence.lower())

    def test_moments_use_the_segment_end_as_the_pause_time(self) -> None:
        moments = segments_to_moments(
            [PeriodicSegment(0.0, 10.0, "A person walks."), PeriodicSegment(10.0, 16.4, "A person sits.")]
        )
        self.assertEqual([moment.timestamp_seconds for moment in moments], [10.0, 16.4])


class ModeSeparationTests(unittest.TestCase):
    def test_speech_scheduler_is_not_required_for_periodic_clips(self) -> None:
        timeline = SpeechTimeline(
            duration_seconds=20.0,
            speech_intervals=(TimeInterval(0.0, 20.0),),
        )
        speech_aware = schedule_narration_clips(
            [NarrationClip(10.0, "A person walks.", clip_id="a" * 32, duration_seconds=2.0)],
            timeline,
        )
        self.assertIsNone(speech_aware[0].scheduled_start_seconds)

        from app.main import _periodic_summary_payload

        periodic = _periodic_summary_payload(
            [PeriodicSegment(0.0, 10.0, "A person walks.")],
            [NarrationClip(10.0, "A person walks.", clip_id="b" * 32, duration_seconds=2.0)],
        )
        self.assertEqual(periodic[0].segment_end_seconds, 10.0)
        self.assertEqual(periodic[0].audio_url, f"/api/narration/{'b' * 32}")
        self.assertEqual(periodic[0].duration_seconds, 2.0)

    def test_missing_piper_clip_keeps_text_without_audio_url(self) -> None:
        from app.main import _periodic_summary_payload

        payload = _periodic_summary_payload(
            [PeriodicSegment(0.0, 8.0, "A person stands.")],
            [NarrationClip(8.0, "A person stands.")],
        )
        self.assertEqual(payload[0].description, "A person stands.")
        self.assertIsNone(payload[0].audio_url)
        self.assertIsNone(payload[0].duration_seconds)

    def test_caption_response_keeps_speech_aware_fields_when_periodic_list_is_empty(self) -> None:
        response = CaptionResponse(
            caption="A person walks.",
            device="cpu",
            sampled_frames=1,
            frame_captions=[
                {
                    "timestamp_seconds": 0.0,
                    "caption": "a person walking",
                    "summary_status": "main_scene",
                }
            ],
            audio_descriptions=[
                AudioDescription(timestamp_seconds=0.0, description="A person is walking.")
            ],
        )
        payload = response.model_dump()
        self.assertEqual(payload["periodic_summaries"], [])
        self.assertEqual(payload["audio_descriptions"][0]["description"], "A person is walking.")
        self.assertIn("scheduled_start_seconds", payload["audio_descriptions"][0])


class PeriodicPlayerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (Path(__file__).resolve().parent.parent / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        self.html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(
            encoding="utf-8"
        )

    def test_mode_selector_defaults_to_speech_aware(self) -> None:
        self.assertIn('option value="speech-aware" selected', self.html)
        self.assertIn('option value="periodic"', self.html)

    def test_periodic_hold_does_not_use_the_speech_aware_pause_path(self) -> None:
        pause_handler = re.search(
            r'videoPreview\.addEventListener\("pause", \(\) => \{(.*?)\}\);',
            self.source,
            flags=re.S,
        )
        self.assertIsNotNone(pause_handler)
        self.assertIn("if (periodicHold) return;", pause_handler.group(1))

    def test_speech_aware_due_clips_still_require_a_scheduled_start(self) -> None:
        self.assertIn(
            "clip.audio_url && clip.scheduled_start_seconds != null && clip.duration_seconds > 0",
            self.source,
        )
        self.assertIn("if (isPeriodicMode()) return;", self.source)
        self.assertIn("playDuePeriodicSummary", self.source)

    def test_periodic_playback_pauses_then_resumes_the_video(self) -> None:
        self.assertIn("videoPreview.pause();", self.source)
        self.assertIn("finishPeriodicHold", self.source)
        self.assertNotIn("schedule_narration_clips", self.source)


if __name__ == "__main__":
    unittest.main()
