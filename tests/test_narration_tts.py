"""Focused tests for local Piper narration clips and persistent clip serving."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np
from fastapi.testclient import TestClient

from app.caption_service import NarrationMoment
from app.main import NARRATION_DIR, app
from app.narration_tts import (
    DEFAULT_VOICE_CONFIG,
    DEFAULT_VOICE_MODEL,
    NarrationClip,
    PiperNarrationSynthesizer,
    resolve_clip_path,
    wav_duration_seconds,
)
from app.schemas import AudioDescription, CaptionResponse


def write_silence_wav(path: Path, seconds: float = 0.5, sample_rate: int = 22050) -> None:
    samples = np.zeros(int(sample_rate * seconds), dtype=np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


class WavDurationTests(unittest.TestCase):
    def test_duration_matches_written_silence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tts-duration-") as temp_dir:
            wav_path = Path(temp_dir) / "silence.wav"
            write_silence_wav(wav_path, seconds=0.4, sample_rate=22050)
            self.assertAlmostEqual(wav_duration_seconds(wav_path), 0.4, places=2)


class MissingModelTests(unittest.TestCase):
    def test_missing_voice_keeps_text_and_skips_audio(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tts-missing-") as temp_dir:
            synthesizer = PiperNarrationSynthesizer(
                model_path=Path(temp_dir) / "missing.onnx",
                config_path=Path(temp_dir) / "missing.onnx.json",
            )
            clips = synthesizer.synthesize_moments(
                [NarrationMoment(timestamp_seconds=1.5, description="A person walks.")],
                Path(temp_dir) / "clips",
            )

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].description, "A person walks.")
        self.assertEqual(clips[0].timestamp_seconds, 1.5)
        self.assertIsNone(clips[0].clip_id)
        self.assertIsNone(clips[0].path)
        self.assertIsNone(clips[0].duration_seconds)
        self.assertFalse(synthesizer.model_files_available())


class ClipPathTests(unittest.TestCase):
    def test_resolve_clip_rejects_path_traversal_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tts-resolve-") as temp_dir:
            directory = Path(temp_dir)
            self.assertIsNone(resolve_clip_path(directory, "../secret"))
            self.assertIsNone(resolve_clip_path(directory, "not-a-clip-id"))
            self.assertIsNone(resolve_clip_path(directory, "a" * 32))

            clip_id = "b" * 32
            wav_path = directory / f"{clip_id}.wav"
            write_silence_wav(wav_path)
            resolved = resolve_clip_path(directory, clip_id)
            self.assertEqual(resolved, wav_path.resolve())


class SchemaCompatibilityTests(unittest.TestCase):
    def test_existing_caption_fields_remain_valid_without_audio(self) -> None:
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
        self.assertIn("caption", payload)
        self.assertIn("frame_captions", payload)
        self.assertEqual(payload["audio_descriptions"][0]["audio_url"], None)
        self.assertEqual(payload["audio_descriptions"][0]["description"], "A person is walking.")


class NarrationApiTests(unittest.TestCase):
    def setUp(self) -> None:
        NARRATION_DIR.mkdir(parents=True, exist_ok=True)
        self.client = TestClient(app)

    def test_unknown_clip_returns_404(self) -> None:
        response = self.client.get(f"/api/narration/{'c' * 32}")
        self.assertEqual(response.status_code, 404)

    def test_existing_clip_is_served_from_managed_directory(self) -> None:
        clip_id = "d" * 32
        wav_path = NARRATION_DIR / f"{clip_id}.wav"
        write_silence_wav(wav_path, seconds=0.2)
        try:
            response = self.client.get(f"/api/narration/{clip_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "audio/wav")
            self.assertGreater(len(response.content), 44)
            self.assertTrue(wav_path.is_file())
        finally:
            wav_path.unlink(missing_ok=True)


@unittest.skipUnless(
    DEFAULT_VOICE_MODEL.is_file() and DEFAULT_VOICE_CONFIG.is_file(),
    "Local Piper voice files are not present.",
)
class PiperSynthesisTests(unittest.TestCase):
    def test_piper_writes_wav_with_duration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tts-piper-") as temp_dir:
            output_dir = Path(temp_dir) / "clips"
            synthesizer = PiperNarrationSynthesizer()
            clips = synthesizer.synthesize_moments(
                [NarrationMoment(timestamp_seconds=2.0, description="A person stands.")],
                output_dir,
            )
            self.assertEqual(len(clips), 1)
            clip = clips[0]
            self.assertIsInstance(clip, NarrationClip)
            self.assertIsNotNone(clip.clip_id)
            self.assertIsNotNone(clip.path)
            self.assertTrue(clip.path.is_file())
            self.assertGreater(clip.duration_seconds or 0.0, 0.2)
            self.assertGreater(clip.path.stat().st_size, 44)
            self.assertNotIn("video-caption-", str(clip.path))


if __name__ == "__main__":
    unittest.main()
