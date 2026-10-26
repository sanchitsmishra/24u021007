from pydantic import BaseModel, Field


class FrameCaption(BaseModel):
    """Caption generated for one sampled moment in the uploaded video."""

    timestamp_seconds: float = Field(ge=0)
    caption: str
    summary_status: str = Field(
        description="Whether this frame supports the main scene, is another scene, or is an outro."
    )


class AudioDescription(BaseModel):
    """A visual description intended to be spoken when the video reaches this moment."""

    timestamp_seconds: float = Field(ge=0)
    description: str
    audio_url: str | None = Field(
        default=None,
        description="URL of the locally generated Piper WAV clip, if synthesis succeeded.",
    )
    duration_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Length of the Piper WAV clip in seconds, if synthesis succeeded.",
    )
    scheduled_start_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Video time when this clip may start without overlapping speech.",
    )


class PeriodicSummary(BaseModel):
    """A 10-second visual summary spoken after the video pauses at segment_end_seconds."""

    timestamp_seconds: float = Field(ge=0, description="Pause time; same as segment_end_seconds.")
    segment_start_seconds: float = Field(ge=0)
    segment_end_seconds: float = Field(ge=0)
    description: str
    audio_url: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)


class CaptionResponse(BaseModel):
    """Result returned after a video has been visually captioned."""

    caption: str
    device: str
    sampled_frames: int = Field(ge=1)
    frame_captions: list[FrameCaption]
    audio_descriptions: list[AudioDescription]
    periodic_summaries: list[PeriodicSummary] = Field(default_factory=list)
