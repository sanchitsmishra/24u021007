"""FastAPI application for the Video Caption Generator prototype."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.caption_service import VideoCaptionService, VideoProcessingError
from app.narration_scheduler import schedule_narration_clips
from app.narration_tts import NarrationClip, PiperNarrationSynthesizer, resolve_clip_path
from app.periodic_summary import (
    PeriodicSegment,
    build_periodic_segments,
    segments_to_moments,
)
from app.schemas import AudioDescription, CaptionResponse, FrameCaption, PeriodicSummary
from app.speech_detector import SpeechDetector, SpeechDetectionError, SpeechTimeline


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
# Request uploads are deleted with the caption tempfile. Piper clips live here so
# the browser can still fetch them after /api/caption returns.
NARRATION_DIR = BASE_DIR / "narration_clips"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

app = FastAPI(title="Video Caption Generator", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
caption_service = VideoCaptionService()
narration_synthesizer = PiperNarrationSynthesizer()
speech_detector = SpeechDetector()


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": caption_service.device}


@app.post("/api/caption", response_model=CaptionResponse)
async def generate_caption(video: UploadFile = File(...)) -> CaptionResponse:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Upload an MP4, MOV, AVI, MKV, or WEBM video file.",
        )

    with tempfile.TemporaryDirectory(prefix="video-caption-") as temp_dir:
        upload_path = Path(temp_dir) / f"upload{suffix}"
        uploaded_bytes = 0
        with upload_path.open("wb") as output:
            while chunk := await video.read(1024 * 1024):
                uploaded_bytes += len(chunk)
                if uploaded_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video must be 200 MB or smaller.")
                output.write(chunk)

        try:
            caption, frame_captions, audio_descriptions = caption_service.caption_video(upload_path)
            speech_timeline = _detect_speech_timeline(upload_path)
            periodic_segments = _build_periodic_segments(upload_path)
        except VideoProcessingError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:  # Keep model/runtime details out of the browser response.
            raise HTTPException(
                status_code=500,
                detail="Caption generation failed. Check the server terminal for details.",
            ) from error
        finally:
            await video.close()

    narration_clips = _synthesize_narration_clips(audio_descriptions)
    scheduled_clips = schedule_narration_clips(narration_clips, speech_timeline)
    periodic_clips = _synthesize_narration_clips(segments_to_moments(periodic_segments))
    return CaptionResponse(
        caption=caption,
        device=caption_service.device,
        sampled_frames=len(frame_captions),
        frame_captions=[
            FrameCaption(
                timestamp_seconds=frame.timestamp_seconds,
                caption=frame.caption,
                summary_status=frame.summary_status,
            )
            for frame in frame_captions
        ],
        audio_descriptions=[
            AudioDescription(
                timestamp_seconds=clip.timestamp_seconds,
                description=clip.description,
                audio_url=f"/api/narration/{clip.clip_id}" if clip.clip_id else None,
                duration_seconds=clip.duration_seconds,
                scheduled_start_seconds=clip.scheduled_start_seconds,
            )
            for clip in scheduled_clips
        ],
        periodic_summaries=_periodic_summary_payload(periodic_segments, periodic_clips),
    )


@app.get("/api/narration/{clip_id}")
def get_narration_clip(clip_id: str) -> FileResponse:
    clip_path = resolve_clip_path(NARRATION_DIR, clip_id)
    if clip_path is None:
        raise HTTPException(status_code=404, detail="Narration clip not found.")
    return FileResponse(clip_path, media_type="audio/wav", filename=f"{clip_id}.wav")


def _build_periodic_segments(upload_path: Path) -> list[PeriodicSegment]:
    """10-second visual summaries; failures must not fail speech-aware captions."""
    try:
        return build_periodic_segments(caption_service, upload_path)
    except Exception:
        LOGGER.exception("Periodic visual summaries failed.")
        return []


def _periodic_summary_payload(
    segments: list[PeriodicSegment], clips: list[NarrationClip]
) -> list[PeriodicSummary]:
    """Attach Piper URLs by index. These clips are never speech-gap scheduled."""
    payload: list[PeriodicSummary] = []
    for index, segment in enumerate(segments):
        clip = clips[index] if index < len(clips) else None
        payload.append(
            PeriodicSummary(
                timestamp_seconds=segment.segment_end_seconds,
                segment_start_seconds=segment.segment_start_seconds,
                segment_end_seconds=segment.segment_end_seconds,
                description=segment.description if clip is None else clip.description,
                audio_url=f"/api/narration/{clip.clip_id}" if clip and clip.clip_id else None,
                duration_seconds=None if clip is None else clip.duration_seconds,
            )
        )
    return payload


def _detect_speech_timeline(upload_path: Path) -> SpeechTimeline:
    """VAD must run while the uploaded file still exists. Failures yield an empty timeline."""
    try:
        return speech_detector.analyze(upload_path)
    except (SpeechDetectionError, Exception):
        LOGGER.exception("Speech detection failed; narration will use speech-free scheduling only.")
        return SpeechTimeline(duration_seconds=0.0, speech_intervals=())


def _synthesize_narration_clips(audio_descriptions):
    """Create Piper WAVs after the upload tempfile is gone; keep caption text on failure."""
    try:
        return narration_synthesizer.synthesize_moments(audio_descriptions, NARRATION_DIR)
    except Exception:
        LOGGER.exception("Piper narration synthesis failed.")
        return [
            NarrationClip(
                timestamp_seconds=moment.timestamp_seconds,
                description=moment.description,
            )
            for moment in audio_descriptions
        ]
