"""Local Piper TTS for visual-narration clips.

This module only turns existing NarrationMoment text into WAV files. Speech-gap
scheduling and playback/volume behaviour stay in later steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import logging
import re
import time
import uuid
import wave

from app.caption_service import NarrationMoment

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VOICE_MODEL = BASE_DIR / "models" / "piper" / "en_US-lessac-medium.onnx"
DEFAULT_VOICE_CONFIG = BASE_DIR / "models" / "piper" / "en_US-lessac-medium.onnx.json"
CLIP_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
DEFAULT_CLIP_MAX_AGE_SECONDS = 24 * 60 * 60


class NarrationTTSError(RuntimeError):
    """Raised when Piper cannot be prepared for synthesis."""


@dataclass(frozen=True)
class NarrationClip:
    """One spoken description plus optional persistent WAV metadata."""

    timestamp_seconds: float
    description: str
    clip_id: str | None = None
    path: Path | None = None
    duration_seconds: float | None = None


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    if sample_rate <= 0 or frame_count <= 0:
        return 0.0
    return round(frame_count / sample_rate, 3)


def prune_old_clips(
    directory: Path, max_age_seconds: float = DEFAULT_CLIP_MAX_AGE_SECONDS
) -> None:
    """Remove leftover narration WAVs so the managed directory cannot grow forever."""
    if not directory.is_dir():
        return
    now = time.time()
    for clip_path in directory.glob("*.wav"):
        try:
            if now - clip_path.stat().st_mtime > max_age_seconds:
                clip_path.unlink()
        except OSError:
            continue


def resolve_clip_path(directory: Path, clip_id: str) -> Path | None:
    """Return a WAV inside directory only when clip_id is a generated identifier."""
    if not CLIP_ID_PATTERN.fullmatch(clip_id):
        return None
    clip_path = (directory / f"{clip_id}.wav").resolve()
    try:
        clip_path.relative_to(directory.resolve())
    except ValueError:
        return None
    if not clip_path.is_file():
        return None
    return clip_path


class PiperNarrationSynthesizer:
    """Loads the project Piper voice once and writes WAV clips next to captions."""

    def __init__(
        self,
        model_path: Path = DEFAULT_VOICE_MODEL,
        config_path: Path = DEFAULT_VOICE_CONFIG,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self._voice = None
        self._load_lock = Lock()
        self._load_failed = False

    def model_files_available(self) -> bool:
        return self.model_path.is_file() and self.config_path.is_file()

    def _ensure_voice_loaded(self) -> bool:
        if self._voice is not None:
            return True
        if self._load_failed:
            return False
        with self._load_lock:
            if self._voice is not None:
                return True
            if self._load_failed:
                return False
            if not self.model_files_available():
                LOGGER.warning(
                    "Piper voice files are missing: %s / %s",
                    self.model_path,
                    self.config_path,
                )
                self._load_failed = True
                return False
            try:
                from piper import PiperVoice

                self._voice = PiperVoice.load(
                    self.model_path,
                    config_path=self.config_path,
                    use_cuda=False,
                )
            except Exception:
                LOGGER.exception("Could not load the local Piper voice.")
                self._load_failed = True
                return False
            return True

    def synthesize_moments(
        self, moments: list[NarrationMoment], output_dir: Path
    ) -> list[NarrationClip]:
        """Write one WAV per description. Failures skip audio and keep the text."""
        output_dir.mkdir(parents=True, exist_ok=True)
        prune_old_clips(output_dir)

        if not self._ensure_voice_loaded():
            return [
                NarrationClip(
                    timestamp_seconds=moment.timestamp_seconds,
                    description=moment.description,
                )
                for moment in moments
            ]

        return [self._synthesize_moment(moment, output_dir) for moment in moments]

    def _synthesize_moment(self, moment: NarrationMoment, output_dir: Path) -> NarrationClip:
        description = moment.description.strip()
        if not description:
            return NarrationClip(
                timestamp_seconds=moment.timestamp_seconds,
                description=moment.description,
            )

        clip_id = uuid.uuid4().hex
        clip_path = output_dir / f"{clip_id}.wav"
        try:
            with wave.open(str(clip_path), "wb") as wav_file:
                self._voice.synthesize_wav(description, wav_file)
            if not clip_path.is_file() or clip_path.stat().st_size <= 44:
                raise NarrationTTSError("Piper wrote an empty WAV file.")
            return NarrationClip(
                timestamp_seconds=moment.timestamp_seconds,
                description=moment.description,
                clip_id=clip_id,
                path=clip_path,
                duration_seconds=wav_duration_seconds(clip_path),
            )
        except Exception:
            LOGGER.exception("Piper could not synthesise a narration clip.")
            try:
                clip_path.unlink(missing_ok=True)
            except OSError:
                pass
            return NarrationClip(
                timestamp_seconds=moment.timestamp_seconds,
                description=moment.description,
            )
