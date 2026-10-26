"""Local speech-interval detection using Silero VAD.

This module only finds where speech is present or absent. Narration generation,
scheduling, and volume ducking are intentionally left to later steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import subprocess
import tempfile
import wave

import cv2
import numpy as np
import torch

SAMPLE_RATE = 16_000
VAD_WINDOW_SAMPLES = 512
DEFAULT_MIN_SILENCE_GAP_SECONDS = 0.20


class SpeechDetectionError(ValueError):
    """Raised when a media file cannot be analysed for speech."""


@dataclass(frozen=True)
class TimeInterval:
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return round(self.end_seconds - self.start_seconds, 3)


@dataclass(frozen=True)
class SpeechTimeline:
    """Speech and speech-free intervals for one media file."""

    duration_seconds: float
    speech_intervals: tuple[TimeInterval, ...]

    def speech_free_intervals(
        self, min_duration_seconds: float = 0.0
    ) -> tuple[TimeInterval, ...]:
        """Return gaps that contain no detected speech.

        Gaps shorter than min_duration_seconds are omitted so later narration
        scheduling can ignore pauses that are too brief to speak in.
        """
        return speech_free_intervals(
            self.duration_seconds,
            self.speech_intervals,
            min_duration_seconds=min_duration_seconds,
        )


def merge_speech_intervals(
    intervals: list[TimeInterval],
    min_silence_gap_seconds: float = DEFAULT_MIN_SILENCE_GAP_SECONDS,
) -> tuple[TimeInterval, ...]:
    """Combine overlapping or closely spaced speech segments into one interval."""
    if not intervals:
        return ()

    ordered = sorted(intervals, key=lambda interval: (interval.start_seconds, interval.end_seconds))
    merged = [ordered[0]]
    for interval in ordered[1:]:
        current = merged[-1]
        gap = interval.start_seconds - current.end_seconds
        if gap <= min_silence_gap_seconds:
            merged[-1] = TimeInterval(
                start_seconds=current.start_seconds,
                end_seconds=max(current.end_seconds, interval.end_seconds),
            )
            continue
        merged.append(interval)
    return tuple(merged)


def speech_free_intervals(
    duration_seconds: float,
    speech_intervals: tuple[TimeInterval, ...] | list[TimeInterval],
    min_duration_seconds: float = 0.0,
) -> tuple[TimeInterval, ...]:
    """Derive speech-free windows from a duration and detected speech intervals."""
    if duration_seconds < 0:
        raise SpeechDetectionError("Media duration cannot be negative.")

    gaps: list[TimeInterval] = []
    cursor = 0.0
    for interval in merge_speech_intervals(list(speech_intervals)):
        start = max(0.0, min(interval.start_seconds, duration_seconds))
        end = max(0.0, min(interval.end_seconds, duration_seconds))
        if start > cursor:
            gaps.append(TimeInterval(start_seconds=round(cursor, 3), end_seconds=round(start, 3)))
        cursor = max(cursor, end)
    if cursor < duration_seconds:
        gaps.append(
            TimeInterval(start_seconds=round(cursor, 3), end_seconds=round(duration_seconds, 3))
        )

    return tuple(
        gap for gap in gaps if gap.duration_seconds >= min_duration_seconds and gap.duration_seconds > 0
    )


def build_speech_timeline(
    duration_seconds: float,
    raw_timestamps: list[dict[str, float]],
    min_silence_gap_seconds: float = DEFAULT_MIN_SILENCE_GAP_SECONDS,
) -> SpeechTimeline:
    """Convert Silero-style {start, end} dicts into a SpeechTimeline."""
    intervals = [
        TimeInterval(
            start_seconds=round(float(item["start"]), 3),
            end_seconds=round(float(item["end"]), 3),
        )
        for item in raw_timestamps
        if float(item["end"]) > float(item["start"])
    ]
    return SpeechTimeline(
        duration_seconds=round(duration_seconds, 3),
        speech_intervals=merge_speech_intervals(intervals, min_silence_gap_seconds),
    )


class SpeechDetector:
    """Loads Silero VAD once per process and detects speech intervals locally."""

    def __init__(self) -> None:
        self._model = None
        self._load_lock = Lock()

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from silero_vad import load_silero_vad
            except ImportError as error:
                raise SpeechDetectionError(
                    "Silero VAD is not installed. Run setup again to install local speech detection."
                ) from error
            self._model = load_silero_vad()

    def analyze(self, media_path: Path) -> SpeechTimeline:
        """Return speech and speech-free intervals for a local video or audio file."""
        path = Path(media_path)
        if not path.is_file():
            raise SpeechDetectionError("The media file could not be found.")

        waveform, duration_seconds = self._load_waveform(path)
        if duration_seconds <= 0 or waveform.numel() == 0:
            return SpeechTimeline(duration_seconds=0.0, speech_intervals=())

        self._ensure_model_loaded()
        from silero_vad import get_speech_timestamps

        if waveform.numel() < VAD_WINDOW_SAMPLES:
            waveform = torch.nn.functional.pad(
                waveform, (0, VAD_WINDOW_SAMPLES - waveform.numel())
            )

        raw_timestamps = get_speech_timestamps(
            waveform,
            self._model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
        )
        return build_speech_timeline(duration_seconds, raw_timestamps)

    def _load_waveform(self, media_path: Path) -> tuple[torch.Tensor, float]:
        if media_path.suffix.lower() == ".wav":
            waveform, sample_rate, duration_seconds = _read_wav_mono_float(media_path)
            if sample_rate == SAMPLE_RATE:
                return waveform, duration_seconds

        with tempfile.TemporaryDirectory(prefix="speech-vad-") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            extracted = _extract_wav_with_ffmpeg(media_path, wav_path)
            if extracted:
                waveform, _sample_rate, duration_seconds = _read_wav_mono_float(wav_path)
                return waveform, duration_seconds

        duration_seconds = _visual_duration_seconds(media_path)
        return torch.zeros(0, dtype=torch.float32), duration_seconds


def _read_wav_mono_float(wav_path: Path) -> tuple[torch.Tensor, int, float]:
    with wave.open(str(wav_path), "rb") as wav_file:
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise SpeechDetectionError("Only 16-bit or 32-bit PCM WAV files are supported.")

    if channel_count > 1:
        audio = audio.reshape(-1, channel_count).mean(axis=1)
    duration_seconds = 0.0 if sample_rate <= 0 else len(audio) / sample_rate
    return torch.from_numpy(np.copy(audio)), sample_rate, duration_seconds


def _extract_wav_with_ffmpeg(media_path: Path, wav_path: Path) -> bool:
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise SpeechDetectionError(
            "imageio-ffmpeg is not installed. Run setup again to extract audio locally."
        ) from error

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        creationflags=creationflags,
    )
    return completed.returncode == 0 and wav_path.is_file() and wav_path.stat().st_size > 0


def _visual_duration_seconds(media_path: Path) -> float:
    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        return 0.0
    try:
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or fps <= 0:
            return 0.0
        return frame_count / fps
    finally:
        capture.release()


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Detect speech and speech-free intervals in a local video or WAV file."
    )
    parser.add_argument("media_path", type=Path)
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.0,
        help="Ignore speech-free gaps shorter than this many seconds.",
    )
    args = parser.parse_args()
    timeline = SpeechDetector().analyze(args.media_path)
    payload = {
        "duration_seconds": timeline.duration_seconds,
        "speech_intervals": [
            {"start_seconds": item.start_seconds, "end_seconds": item.end_seconds}
            for item in timeline.speech_intervals
        ],
        "speech_free_intervals": [
            {"start_seconds": item.start_seconds, "end_seconds": item.end_seconds}
            for item in timeline.speech_free_intervals(args.min_gap)
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
