const form = document.querySelector("#caption-form");
const fileInput = document.querySelector("#video-input");
const fileLabel = document.querySelector("#file-label");
const submitButton = document.querySelector("#submit-button");
const status = document.querySelector("#status");
const result = document.querySelector("#result");
const captionHeading = document.querySelector("#caption-heading");
const meta = document.querySelector("#meta");
const frameCaptions = document.querySelector("#frame-captions");
const dropZone = document.querySelector("#drop-zone");
const downloadButton = document.querySelector("#download-button");
const historySection = document.querySelector("#history");
const historyList = document.querySelector("#history-list");
const clearHistoryButton = document.querySelector("#clear-history");
const videoPlayerSection = document.querySelector("#video-player-section");
const videoPreview = document.querySelector("#video-preview");
const narrationButton = document.querySelector("#narration-button");
const narrationRate = document.querySelector("#narration-rate");
const narrationMode = document.querySelector("#narration-mode");
const narrationNote = document.querySelector("#narration-note");
const audioDescriptions = document.querySelector("#audio-descriptions");
const periodicSummariesList = document.querySelector("#periodic-summaries");
const SPEECH_AWARE_NOTE = "Repeated views are combined, while meaningful new actions and visual changes are narrated directly. Descriptions play only in speech-free gaps. The video keeps playing; its volume is lowered during each clip, then restored.";
const PERIODIC_NOTE = "Every 10 seconds the video pauses for a concise visual summary, then resumes automatically. This mode ignores dialogue so summaries still play during continuous speech.";
const HISTORY_KEY = "video-caption-generator-history-v1";
const NARRATION_TRIGGER_SECONDS = 0.35;
const DUCKED_VOLUME = 0.2;
const narrationAudio = new Audio();
narrationAudio.preload = "auto";
let activeDescriptions = [];
let spokenDescriptionIndexes = new Set();
let narrationEnabled = false;
let previewUrl = null;
let ducked = false;
let volumeBeforeDuck = null;
let userVolume = 1;
let activeNarrationIndex = null;
let periodicSummaries = [];
let periodicHold = false;

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function renderHistory() {
  const entries = getHistory();
  historyList.replaceChildren(
    ...entries.map(({ caption, createdAt }) => {
      const item = document.createElement("li");
      const date = new Date(createdAt).toLocaleString();
      item.textContent = `${caption} — ${date}`;
      return item;
    }),
  );
  historySection.classList.toggle("hidden", entries.length === 0);
}

function saveToHistory(data) {
  const entries = getHistory();
  entries.unshift({
    caption: data.caption,
    createdAt: new Date().toISOString(),
  });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, 8)));
  renderHistory();
}

function isPeriodicMode() {
  return narrationMode && narrationMode.value === "periodic";
}

function isPlayableClip(clip) {
  return Boolean(clip && clip.audio_url && clip.scheduled_start_seconds != null && clip.duration_seconds > 0);
}

function isPlayablePeriodicClip(clip) {
  return Boolean(clip && clip.audio_url && clip.duration_seconds > 0 && clip.segment_end_seconds != null);
}

function playableClipsForMode() {
  return isPeriodicMode() ? periodicSummaries.filter(isPlayablePeriodicClip) : activeDescriptions.filter(isPlayableClip);
}

function updateModeUi() {
  if (narrationNote) {
    narrationNote.textContent = isPeriodicMode() ? PERIODIC_NOTE : SPEECH_AWARE_NOTE;
  }
  narrationButton.disabled = playableClipsForMode().length === 0;
}

function clipStart(clip) {
  return clip.scheduled_start_seconds;
}

function restoreVideoVolume() {
  if (!ducked) return;
  videoPreview.volume = volumeBeforeDuck;
  ducked = false;
  volumeBeforeDuck = null;
}

function duckVideoVolume() {
  if (ducked) return;
  volumeBeforeDuck = videoPreview.volume;
  ducked = true;
  videoPreview.volume = Math.min(volumeBeforeDuck, DUCKED_VOLUME);
}

function shouldResumePausedNarration(state) {
  return Boolean(
    state.enabled &&
      state.activeIndex != null &&
      state.hasSource &&
      !state.ended,
  );
}

function shouldIgnoreNarrationPlayError(error) {
  return Boolean(error && error.name === "AbortError");
}

function narrationHasSource() {
  const src = narrationAudio.currentSrc || narrationAudio.getAttribute("src") || "";
  return src.length > 0;
}

function pauseNarrationAudio() {
  if (!narrationAudio.paused) {
    narrationAudio.pause();
  }
}

function stopNarrationAudio() {
  narrationAudio.pause();
  narrationAudio.removeAttribute("src");
  narrationAudio.load();
  activeNarrationIndex = null;
  restoreVideoVolume();
}

function playNarrationAudio() {
  const playAttempt = narrationAudio.play();
  if (playAttempt && typeof playAttempt.catch === "function") {
    playAttempt.catch((error) => {
      if (shouldIgnoreNarrationPlayError(error)) return;
      stopNarrationAudio();
    });
  }
}

function resumePausedNarration() {
  if (
    !shouldResumePausedNarration({
      enabled: narrationEnabled,
      activeIndex: activeNarrationIndex,
      hasSource: narrationHasSource(),
      ended: narrationAudio.ended,
    })
  ) {
    return false;
  }
  playNarrationAudio();
  return true;
}

function markPastClipsSpoken() {
  spokenDescriptionIndexes = new Set(
    activeDescriptions
      .map((clip, index) => {
        if (index === activeNarrationIndex) return null;
        if (!isPlayableClip(clip)) return index;
        return clipStart(clip) < videoPreview.currentTime - NARRATION_TRIGGER_SECONDS ? index : null;
      })
      .filter((index) => index !== null),
  );
}

function showSelectedFile(file) {
  fileLabel.textContent = file ? file.name : "Choose a video";
  stopNarration();
  narrationButton.disabled = true;
  activeDescriptions = [];
  periodicSummaries = [];
  spokenDescriptionIndexes = new Set();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  if (file) {
    previewUrl = URL.createObjectURL(file);
    videoPreview.src = previewUrl;
    videoPlayerSection.classList.remove("hidden");
  } else {
    videoPreview.removeAttribute("src");
    videoPlayerSection.classList.add("hidden");
  }
}

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function showResult(data) {
  captionHeading.textContent = data.caption;
  meta.textContent = `${data.sampled_frames} visual moments analysed · ${data.audio_descriptions.length} essential descriptions · running on ${data.device.toUpperCase()}`;
  frameCaptions.replaceChildren(
    ...data.frame_captions.map(({ timestamp_seconds: seconds, caption, summary_status: summaryStatus }) => {
      const item = document.createElement("li");
      const label = {
        main_scene: "main visual scene",
        additional_scene: "additional visual scene",
        ignored: "ignored: likely outro/logo",
      }[summaryStatus] || "detected visual scene";
      item.textContent = `${seconds.toFixed(2)}s — ${caption} (${label})`;
      item.classList.toggle("key-moment", summaryStatus === "main_scene");
      item.classList.toggle("additional-scene", summaryStatus === "additional_scene");
      return item;
    }),
  );
  activeDescriptions = data.audio_descriptions || [];
  periodicSummaries = data.periodic_summaries || [];
  spokenDescriptionIndexes = new Set();
  audioDescriptions.replaceChildren(
    ...activeDescriptions.map((clip) => {
      const item = document.createElement("li");
      const start = clip.scheduled_start_seconds ?? clip.timestamp_seconds;
      const skipped = isPlayableClip(clip) ? "" : " (text only)";
      item.textContent = `${Number(start).toFixed(2)}s — ${clip.description}${skipped}`;
      return item;
    }),
  );
  if (periodicSummariesList) {
    periodicSummariesList.replaceChildren(
      ...periodicSummaries.map((clip) => {
        const item = document.createElement("li");
        const skipped = isPlayablePeriodicClip(clip) ? "" : " (text only)";
        item.textContent = `${Number(clip.segment_start_seconds).toFixed(2)}–${Number(clip.segment_end_seconds).toFixed(2)}s — ${clip.description}${skipped}`;
        return item;
      }),
    );
  }
  updateModeUi();
  result.classList.remove("hidden");
  saveToHistory(data);
}

function stopNarration() {
  narrationEnabled = false;
  periodicHold = false;
  stopNarrationAudio();
  narrationButton.textContent = "Play with audio descriptions";
}

function markPastPeriodicSummaries() {
  spokenDescriptionIndexes = new Set(
    periodicSummaries
      .map((clip, index) => {
        if (index === activeNarrationIndex) return null;
        if (!isPlayablePeriodicClip(clip)) return index;
        return clip.segment_end_seconds < videoPreview.currentTime - NARRATION_TRIGGER_SECONDS ? index : null;
      })
      .filter((index) => index !== null),
  );
}

function finishPeriodicHold(resumeVideo) {
  periodicHold = false;
  activeNarrationIndex = null;
  stopNarrationAudio();
  if (!resumeVideo || !narrationEnabled || videoPreview.ended) {
    if (videoPreview.ended || !narrationEnabled) stopNarration();
    return;
  }
  videoPreview.play().catch(() => {
    setStatus("Press play to continue after the visual summary.", true);
  });
}

function startPeriodicHold(index) {
  const clip = periodicSummaries[index];
  spokenDescriptionIndexes.add(index);
  activeNarrationIndex = index;
  periodicHold = true;
  videoPreview.pause();
  if (!isPlayablePeriodicClip(clip)) {
    finishPeriodicHold(true);
    return;
  }
  narrationAudio.src = clip.audio_url;
  playNarrationAudio();
}

function playDuePeriodicSummary() {
  if (!narrationEnabled || !isPeriodicMode() || periodicHold) return;
  if (videoPreview.paused && !videoPreview.ended) return;
  const nextIndex = periodicSummaries.findIndex(
    (clip, index) =>
      isPlayablePeriodicClip(clip) &&
      !spokenDescriptionIndexes.has(index) &&
      clip.segment_end_seconds <= videoPreview.currentTime + NARRATION_TRIGGER_SECONDS,
  );
  if (nextIndex === -1) return;
  startPeriodicHold(nextIndex);
}

function playDueDescription() {
  if (isPeriodicMode()) return;
  if (!narrationEnabled || videoPreview.paused) return;
  if (activeNarrationIndex !== null) {
    if (narrationAudio.paused) resumePausedNarration();
    return;
  }
  if (!narrationAudio.paused) return;
  const nextIndex = activeDescriptions.findIndex(
    (clip, index) =>
      isPlayableClip(clip) &&
      !spokenDescriptionIndexes.has(index) &&
      clipStart(clip) <= videoPreview.currentTime + NARRATION_TRIGGER_SECONDS &&
      videoPreview.currentTime < clipStart(clip) + clip.duration_seconds,
  );
  if (nextIndex === -1) return;

  const clip = activeDescriptions[nextIndex];
  spokenDescriptionIndexes.add(nextIndex);
  activeNarrationIndex = nextIndex;
  narrationAudio.src = clip.audio_url;
  duckVideoVolume();
  playNarrationAudio();
}

fileInput.addEventListener("change", () => showSelectedFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  showSelectedFile(file);
});

narrationButton.addEventListener("click", async () => {
  if (narrationEnabled) {
    stopNarration();
    return;
  }
  narrationEnabled = true;
  if (isPeriodicMode()) markPastPeriodicSummaries();
  else markPastClipsSpoken();
  narrationButton.textContent = "Stop audio descriptions";
  try {
    await videoPreview.play();
    if (isPeriodicMode()) playDuePeriodicSummary();
    else playDueDescription();
  } catch {
    setStatus("Press play on the video, then turn on audio descriptions.", true);
  }
});

if (narrationMode) {
  narrationMode.addEventListener("change", () => {
    stopNarration();
    updateModeUi();
  });
}

videoPreview.addEventListener("timeupdate", () => {
  if (isPeriodicMode()) playDuePeriodicSummary();
  else playDueDescription();
});
videoPreview.addEventListener("play", () => {
  if (!narrationEnabled) return;
  if (isPeriodicMode()) {
    if (periodicHold) {
      videoPreview.pause();
      if (narrationAudio.paused) playNarrationAudio();
    }
    return;
  }
  if (resumePausedNarration()) return;
  markPastClipsSpoken();
  playDueDescription();
});
videoPreview.addEventListener("seeking", () => {
  periodicHold = false;
  stopNarrationAudio();
});
videoPreview.addEventListener("seeked", () => {
  if (!narrationEnabled) return;
  if (isPeriodicMode()) {
    markPastPeriodicSummaries();
    if (!videoPreview.paused) playDuePeriodicSummary();
    return;
  }
  markPastClipsSpoken();
  if (!videoPreview.paused) playDueDescription();
});
videoPreview.addEventListener("pause", () => {
  if (periodicHold) return;
  pauseNarrationAudio();
});
videoPreview.addEventListener("ended", () => {
  if (isPeriodicMode() && narrationEnabled) {
    playDuePeriodicSummary();
    if (periodicHold) return;
  }
  stopNarration();
});
videoPreview.addEventListener("volumechange", () => {
  if (!ducked) userVolume = videoPreview.volume;
});
narrationAudio.addEventListener("ended", () => {
  if (isPeriodicMode() && periodicHold) {
    const shouldResume = narrationEnabled && !videoPreview.ended;
    finishPeriodicHold(shouldResume);
    return;
  }
  activeNarrationIndex = null;
  restoreVideoVolume();
});
narrationAudio.addEventListener("error", () => {
  if (isPeriodicMode() && periodicHold) {
    finishPeriodicHold(narrationEnabled && !videoPreview.ended);
    return;
  }
  activeNarrationIndex = null;
  restoreVideoVolume();
});

downloadButton.addEventListener("click", () => {
  const frameDetails = [...frameCaptions.querySelectorAll("li")]
    .map((item) => `- ${item.textContent}`)
    .join("\n");
  const narrationScript = [...audioDescriptions.querySelectorAll("li")]
    .map((item) => `- ${item.textContent}`)
    .join("\n");
  const periodicScript = periodicSummariesList
    ? [...periodicSummariesList.querySelectorAll("li")].map((item) => `- ${item.textContent}`).join("\n")
    : "";
  const content = `Video Caption Generator\n\nCaption: ${captionHeading.textContent}\n${meta.textContent}\n\nAudio-description script:\n${narrationScript}\n\nPeriodic visual summaries:\n${periodicScript}\n\nDetected moments:\n${frameDetails}\n`;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([content], { type: "text/plain" }));
  link.download = "video-caption-result.txt";
  link.click();
  URL.revokeObjectURL(link.href);
});

clearHistoryButton.addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const [file] = fileInput.files;
  if (!file) {
    setStatus("Choose a video first.", true);
    return;
  }

  submitButton.disabled = true;
  result.classList.add("hidden");
  setStatus("Reading the video and generating visual captions… First use downloads the AI model.");

  try {
    const body = new FormData();
    body.append("video", file);
    const response = await fetch("/api/caption", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Caption generation failed.");
    showResult(data);
    setStatus("Done.");
  } catch (error) {
    setStatus(error.message || "Something went wrong. Try another short video.", true);
  } finally {
    submitButton.disabled = false;
  }
});

renderHistory();
updateModeUi();
