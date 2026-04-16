from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from speakernotes_whisper.transcribe import (
    TranscriptResult,
    _resolve_device_and_compute,
    load_model,
    transcribe_audio,
)


# ---------------------------------------------------------------------------
# TranscriptResult dataclass
# ---------------------------------------------------------------------------


class TestTranscriptResult:
    def test_fields_are_accessible(self):
        result = TranscriptResult(
            text="Hello world",
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello world"}],
            language="en",
            duration_seconds=1.0,
        )
        assert result.text == "Hello world"
        assert result.segments == [{"start": 0.0, "end": 1.0, "text": "Hello world"}]
        assert result.language == "en"
        assert result.duration_seconds == 1.0

    def test_empty_transcript(self):
        result = TranscriptResult(text="", segments=[], language="auto", duration_seconds=0.0)
        assert result.text == ""
        assert result.segments == []
        assert result.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# _resolve_device_and_compute
# ---------------------------------------------------------------------------


class TestResolveDeviceAndCompute:
    def test_cuda_available_returns_cuda_float16(self):
        mock_ct2 = MagicMock()
        mock_ct2.get_supported_compute_types.return_value = ["cuda", "float16"]
        with patch.dict("sys.modules", {"ctranslate2": mock_ct2}):
            device, compute = _resolve_device_and_compute()
        assert device == "cuda"
        assert compute == "float16"

    def test_cuda_not_in_types_falls_through_to_cpu(self):
        mock_ct2 = MagicMock()
        # First call (cuda check): types without "cuda"; second call (cpu check): non-empty
        mock_ct2.get_supported_compute_types.side_effect = [["float32", "int8"], ["int8"]]
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"ctranslate2": mock_ct2, "torch": mock_torch}):
            device, compute = _resolve_device_and_compute()
        assert device == "cpu"
        assert compute == "int8"

    def test_mps_available_returns_cpu_int8(self):
        mock_ct2 = MagicMock()
        mock_ct2.get_supported_compute_types.side_effect = [["float32"], ["int8"]]
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"ctranslate2": mock_ct2, "torch": mock_torch}):
            device, compute = _resolve_device_and_compute()
        assert device == "cpu"
        assert compute == "int8"

    def test_ctranslate2_unavailable_returns_cpu_int8(self):
        # sys.modules[name] = None makes `import name` raise ImportError
        with patch.dict("sys.modules", {"ctranslate2": None}):
            device, compute = _resolve_device_and_compute()
        assert device == "cpu"
        assert compute == "int8"

    def test_ctranslate2_raises_exception_returns_cpu_int8(self):
        mock_ct2 = MagicMock()
        mock_ct2.get_supported_compute_types.side_effect = RuntimeError("GPU error")
        with patch.dict("sys.modules", {"ctranslate2": mock_ct2}):
            device, compute = _resolve_device_and_compute()
        assert device == "cpu"
        assert compute == "int8"


# ---------------------------------------------------------------------------
# transcribe_audio
# ---------------------------------------------------------------------------


def _make_mock_model(
    segments: list[tuple[float, float, str]],
    duration: float = 10.0,
    language: str = "en",
) -> MagicMock:
    """Build a mock WhisperModel whose transcribe() returns the given segments."""
    mock_info = MagicMock()
    mock_info.duration = duration
    mock_info.language = language

    mock_segs = []
    for start, end, text in segments:
        seg = MagicMock()
        seg.start = start
        seg.end = end
        seg.text = text
        mock_segs.append(seg)

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (mock_segs, mock_info)
    return mock_model


class TestTranscribeAudio:
    def test_returns_transcript_result_instance(self):
        model = _make_mock_model([(0.0, 1.0, " Hello ")])
        assert isinstance(transcribe_audio("test.mp3", model), TranscriptResult)

    def test_text_is_joined_and_stripped_from_segments(self):
        model = _make_mock_model([(0.0, 1.0, "  Hello  "), (1.0, 2.0, "  world  ")])
        result = transcribe_audio("test.mp3", model)
        assert result.text == "Hello world"

    def test_empty_segments_produce_empty_text(self):
        model = _make_mock_model([], duration=0.0)
        result = transcribe_audio("test.mp3", model)
        assert result.text == ""
        assert result.segments == []

    def test_whitespace_only_segments_excluded_from_full_text(self):
        # Whitespace segments strip to "" which is falsy
        model = _make_mock_model([(0.0, 1.0, "  "), (1.0, 2.0, "  real text  ")])
        result = transcribe_audio("test.mp3", model)
        assert result.text == "real text"

    def test_segment_dicts_have_correct_structure(self):
        model = _make_mock_model([(0.5, 2.3, "  Test  "), (2.5, 4.0, "  More  ")])
        result = transcribe_audio("test.mp3", model)
        assert len(result.segments) == 2
        assert result.segments[0] == {"start": 0.5, "end": 2.3, "text": "Test"}
        assert result.segments[1] == {"start": 2.5, "end": 4.0, "text": "More"}

    def test_language_and_duration_come_from_model_info(self):
        model = _make_mock_model([], duration=42.5, language="de")
        result = transcribe_audio("test.mp3", model)
        assert result.language == "de"
        assert result.duration_seconds == 42.5

    def test_progress_callback_called_once_per_segment(self):
        model = _make_mock_model(
            [(0.0, 30.0, "First"), (30.0, 60.0, "Second")], duration=60.0
        )
        calls: list[tuple[float, str]] = []
        transcribe_audio("test.mp3", model, progress_callback=lambda p, m: calls.append((p, m)))
        assert len(calls) == 2

    def test_progress_callback_fraction_matches_segment_position(self):
        model = _make_mock_model([(0.0, 30.0, "Halfway")], duration=60.0)
        fractions: list[float] = []
        transcribe_audio("test.mp3", model, progress_callback=lambda p, m: fractions.append(p))
        assert fractions[0] == pytest.approx(0.5)

    def test_progress_callback_message_contains_elapsed_and_total_minutes(self):
        model = _make_mock_model([(0.0, 90.0, "segment")], duration=180.0)
        msgs: list[str] = []
        transcribe_audio("test.mp3", model, progress_callback=lambda p, m: msgs.append(m))
        # 90s ÷ 60 = 1m; 180s ÷ 60 = 3m
        assert "1m" in msgs[0]
        assert "3m" in msgs[0]

    def test_progress_fraction_capped_at_1(self):
        # segment.end > total_duration (edge case in floating-point)
        model = _make_mock_model([(0.0, 10.1, "over")], duration=10.0)
        fractions: list[float] = []
        transcribe_audio("test.mp3", model, progress_callback=lambda p, m: fractions.append(p))
        assert fractions[0] <= 1.0

    def test_no_progress_callback_runs_without_error(self):
        model = _make_mock_model([(0.0, 1.0, "text")])
        result = transcribe_audio("test.mp3", model, progress_callback=None)
        assert result.text == "text"

    def test_language_parameter_forwarded_to_model(self):
        model = _make_mock_model([], language="fr")
        transcribe_audio("test.mp3", model, language="fr")
        _, kwargs = model.transcribe.call_args
        assert kwargs["language"] == "fr"

    def test_audio_path_forwarded_to_model(self):
        model = _make_mock_model([])
        transcribe_audio("/path/to/audio.wav", model)
        args, _ = model.transcribe.call_args
        assert args[0] == "/path/to/audio.wav"

    def test_vad_filter_enabled(self):
        model = _make_mock_model([])
        transcribe_audio("test.mp3", model)
        _, kwargs = model.transcribe.call_args
        assert kwargs["vad_filter"] is True

    def test_word_timestamps_disabled(self):
        model = _make_mock_model([])
        transcribe_audio("test.mp3", model)
        _, kwargs = model.transcribe.call_args
        assert kwargs["word_timestamps"] is False


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_creates_whisper_model_with_given_size(self):
        mock_whisper_cls = MagicMock()
        mock_fw = MagicMock()
        mock_fw.WhisperModel = mock_whisper_cls

        with patch("speakernotes_whisper.transcribe._resolve_device_and_compute", return_value=("cpu", "int8")):
            with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
                load_model("small")

        mock_whisper_cls.assert_called_once_with("small", device="cpu", compute_type="int8")

    def test_device_and_compute_type_come_from_resolver(self):
        mock_whisper_cls = MagicMock()
        mock_fw = MagicMock()
        mock_fw.WhisperModel = mock_whisper_cls

        with patch("speakernotes_whisper.transcribe._resolve_device_and_compute", return_value=("cuda", "float16")):
            with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
                load_model("large-v3")

        _, kwargs = mock_whisper_cls.call_args
        assert kwargs["device"] == "cuda"
        assert kwargs["compute_type"] == "float16"
