"""Contracts for pausing and resuming an in-progress Piper clip.

The browser player in static/app.js must keep the Audio element's currentTime
on video pause and continue from that position on resume. Volume ducking stays
until the clip finishes, the user seeks, or narration is stopped.
"""

from __future__ import annotations

from pathlib import Path
import re
import unittest

APP_JS = Path(__file__).resolve().parent.parent / "static" / "app.js"


def should_resume_paused_narration(
    enabled: bool, active_index: int | None, has_source: bool, ended: bool
) -> bool:
    return bool(enabled and active_index is not None and has_source and not ended)


def should_ignore_narration_play_error(error_name: str) -> bool:
    return error_name == "AbortError"


def spoken_indexes_after_marking_past(
    clip_starts: list[float],
    current_time: float,
    trigger_seconds: float,
    active_index: int | None,
) -> set[int]:
    spoken: set[int] = set()
    for index, start in enumerate(clip_starts):
        if index == active_index:
            continue
        if start < current_time - trigger_seconds:
            spoken.add(index)
    return spoken


def should_restore_volume(event: str) -> bool:
    return event in {"ended", "error", "seeking", "stopped"}


class PauseResumeDecisionTests(unittest.TestCase):
    def test_resume_continues_the_same_paused_clip(self) -> None:
        self.assertTrue(
            should_resume_paused_narration(
                enabled=True, active_index=0, has_source=True, ended=False
            )
        )

    def test_do_not_resume_after_the_clip_has_finished(self) -> None:
        self.assertFalse(
            should_resume_paused_narration(
                enabled=True, active_index=0, has_source=True, ended=True
            )
        )

    def test_do_not_resume_when_narration_is_disabled_or_cleared(self) -> None:
        self.assertFalse(
            should_resume_paused_narration(
                enabled=False, active_index=0, has_source=True, ended=False
            )
        )
        self.assertFalse(
            should_resume_paused_narration(
                enabled=True, active_index=None, has_source=False, ended=False
            )
        )

    def test_volume_stays_ducked_on_pause_but_restores_on_finish_seek_or_stop(self) -> None:
        self.assertFalse(should_restore_volume("pause"))
        self.assertTrue(should_restore_volume("ended"))
        self.assertTrue(should_restore_volume("seeking"))
        self.assertTrue(should_restore_volume("stopped"))
        self.assertTrue(should_restore_volume("error"))

    def test_interrupted_play_does_not_clear_the_paused_clip(self) -> None:
        self.assertTrue(should_ignore_narration_play_error("AbortError"))
        self.assertFalse(should_ignore_narration_play_error("NotSupportedError"))

    def test_resync_does_not_mark_the_active_paused_clip_as_already_spoken(self) -> None:
        spoken = spoken_indexes_after_marking_past(
            clip_starts=[1.0, 8.0],
            current_time=2.5,
            trigger_seconds=0.35,
            active_index=0,
        )
        self.assertEqual(spoken, set())


class AppJsPauseResumeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = APP_JS.read_text(encoding="utf-8")

    def test_pause_handler_pauses_audio_without_clearing_it(self) -> None:
        pause_handler = re.search(
            r'videoPreview\.addEventListener\("pause", \(\) => \{(.*?)\}\);',
            self.source,
            flags=re.S,
        )
        self.assertIsNotNone(pause_handler)
        body = pause_handler.group(1)
        self.assertIn("pauseNarrationAudio()", body)
        self.assertNotIn("stopNarrationAudio()", body)
        self.assertNotIn("restoreVideoVolume()", body)

    def test_resume_prefers_the_paused_clip_before_scheduling_the_next(self) -> None:
        play_handler = re.search(
            r'videoPreview\.addEventListener\("play", \(\) => \{(.*?)\}\);',
            self.source,
            flags=re.S,
        )
        self.assertIsNotNone(play_handler)
        body = play_handler.group(1)
        resume_at = body.find("resumePausedNarration()")
        mark_at = body.find("markPastClipsSpoken()")
        self.assertNotEqual(resume_at, -1)
        self.assertNotEqual(mark_at, -1)
        self.assertLess(resume_at, mark_at)

    def test_stop_and_seek_still_clear_the_clip_and_restore_volume(self) -> None:
        self.assertIn("function stopNarrationAudio()", self.source)
        self.assertIn("restoreVideoVolume()", self.source)
        seeking_handler = re.search(
            r'videoPreview\.addEventListener\("seeking", \(\) => \{(.*?)\}\);',
            self.source,
            flags=re.S,
        )
        self.assertIsNotNone(seeking_handler)
        self.assertIn("stopNarrationAudio()", seeking_handler.group(1))

    def test_due_clip_lookup_resumes_the_active_clip_instead_of_starting_another(self) -> None:
        self.assertIn("if (activeNarrationIndex !== null) {", self.source)
        self.assertIn("if (narrationAudio.paused) resumePausedNarration();", self.source)
        self.assertIn("shouldIgnoreNarrationPlayError(error)", self.source)
        self.assertIn("if (index === activeNarrationIndex) return null;", self.source)


if __name__ == "__main__":
    unittest.main()
