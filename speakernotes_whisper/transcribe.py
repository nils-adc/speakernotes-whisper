from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class TranscriptResult:
    text: str
    segments: list[dict]  # [{start, end, text}]
    language: str
    duration_seconds: float


def _resolve_device_and_compute() -> tuple[str, str]:
    """Auto-detect the best device/compute_type combo available."""
    try:
        import ctranslate2

        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            return "cuda", "float16"
    except Exception:
        pass

    try:
        import ctranslate2  # noqa: F811

        if ctranslate2.get_supported_compute_types("cpu"):
            # Check for Apple MPS via torch
            try:
                import torch

                if torch.backends.mps.is_available():
                    return "cpu", "int8"  # faster-whisper doesn't support MPS directly; use CPU int8
            except Exception:
                pass
    except Exception:
        pass

    return "cpu", "int8"


def load_model(model_size: str):
    """Load a WhisperModel. Intended to be wrapped with @st.cache_resource in the app."""
    from faster_whisper import WhisperModel

    device, compute_type = _resolve_device_and_compute()
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(
    audio_path: str,
    model,  # WhisperModel instance (cached by caller)
    language: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> TranscriptResult:
    """
    Transcribe an audio/video file using a pre-loaded WhisperModel.

    Args:
        audio_path: Path to the audio/video file.
        model: A loaded faster_whisper.WhisperModel instance.
        language: ISO language code (e.g. "en"), or None for auto-detect.
        progress_callback: Called with (fraction 0–1, status message) as segments complete.

    Returns:
        TranscriptResult with full text, per-segment data, detected language, and duration.
    """
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=False,
    )

    segments: list[dict] = []
    total_duration = info.duration

    for segment in segments_iter:
        segments.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }
        )
        if progress_callback and total_duration > 0:
            pct = min(segment.end / total_duration, 1.0)
            elapsed_min = int(segment.end / 60)
            total_min = int(total_duration / 60)
            progress_callback(pct, f"Transcribed {elapsed_min}m of {total_min}m...")

    full_text = " ".join(s["text"] for s in segments if s["text"])

    return TranscriptResult(
        text=full_text,
        segments=segments,
        language=info.language,
        duration_seconds=total_duration,
    )
