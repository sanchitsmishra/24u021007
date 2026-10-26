"""Tests for local speech detection and speech-interval calculation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from app.speech_detector import (
    SAMPLE_RATE,
    SpeechDetector,
    SpeechTimeline,
    TimeInterval,
    build_speech_timeline,
    merge_speech_intervals,
    speech_free_intervals,
)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    audio = np.clip(samples, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


class SpeechIntervalTests(unittest.TestCase):
    def test_merge_overlaps_and_short_gaps(self) -> None:
        merged = merge_speech_intervals(
            [
                TimeInterval(1.0, 1.8),
                TimeInterval(1.5, 2.2),
                TimeInterval(2.3, 3.0),
                TimeInterval(5.0, 6.0),
            ],
            min_silence_gap_seconds=0.2,
        )
        self.assertEqual(
            merged,
            (
                TimeInterval(1.0, 3.0),
                TimeInterval(5.0, 6.0),
            ),
        )

    def test_speech_free_includes_opening_and_closing_gaps(self) -> None:
        gaps = speech_free_intervals(
            duration_seconds=10.0,
            speech_intervals=(
                TimeInterval(2.0, 4.0),
                TimeInterval(7.0, 8.5),
            ),
        )
        self.assertEqual(
            gaps,
            (
                TimeInterval(0.0, 2.0),
                TimeInterval(4.0, 7.0),
                TimeInterval(8.5, 10.0),
            ),
        )

    def test_no_speech_means_the_whole_file_is_speech_free(self) -> None:
        gaps = speech_free_intervals(duration_seconds=5.0, speech_intervals=())
        self.assertEqual(gaps, (TimeInterval(0.0, 5.0),))

    def test_continuous_speech_has_no_speech_free_interval(self) -> None:
        gaps = speech_free_intervals(
            duration_seconds=5.0,
            speech_intervals=(TimeInterval(0.0, 5.0),),
        )
        self.assertEqual(gaps, ())

    def test_short_gaps_can_be_ignored_for_later_narration(self) -> None:
        gaps = speech_free_intervals(
            duration_seconds=8.0,
            speech_intervals=(
                TimeInterval(1.0, 2.0),
                TimeInterval(2.3, 3.0),
            ),
            min_duration_seconds=1.0,
        )
        self.assertEqual(
            gaps,
            (
                TimeInterval(0.0, 1.0),
                TimeInterval(3.0, 8.0),
            ),
        )

    def test_build_timeline_from_silero_style_timestamps(self) -> None:
        timeline = build_speech_timeline(
            duration_seconds=6.0,
            raw_timestamps=[
                {"start": 1.01, "end": 2.0},
                {"start": 2.05, "end": 2.8},
            ],
            min_silence_gap_seconds=0.2,
        )
        self.assertEqual(timeline.speech_intervals, (TimeInterval(1.01, 2.8),))
        self.assertEqual(
            timeline.speech_free_intervals(),
            (
                TimeInterval(0.0, 1.01),
                TimeInterval(2.8, 6.0),
            ),
        )

    def test_timeline_filters_usable_speech_free_windows(self) -> None:
        timeline = SpeechTimeline(
            duration_seconds=4.0,
            speech_intervals=(TimeInterval(0.4, 3.8),),
        )
        self.assertEqual(timeline.speech_free_intervals(min_duration_seconds=0.5), ())
        self.assertEqual(
            timeline.speech_free_intervals(min_duration_seconds=0.2),
            (TimeInterval(0.0, 0.4), TimeInterval(3.8, 4.0)),
        )


class SpeechDetectorTests(unittest.TestCase):
    def test_silent_wav_has_no_speech_and_one_speech_free_interval(self) -> None:
        detector = SpeechDetector()
        silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        with tempfile.TemporaryDirectory(prefix="vad-test-") as temp_dir:
            wav_path = Path(temp_dir) / "silence.wav"
            write_wav(wav_path, silence)
            timeline = detector.analyze(wav_path)

        self.assertGreaterEqual(timeline.duration_seconds, 1.9)
        self.assertEqual(timeline.speech_intervals, ())
        self.assertEqual(len(timeline.speech_free_intervals()), 1)
        self.assertAlmostEqual(
            timeline.speech_free_intervals()[0].duration_seconds,
            timeline.duration_seconds,
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
