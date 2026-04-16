from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from speakernotes_whisper.claude_client import (
    _MAX_TRANSCRIPT_CHARS,
    SYSTEM_PROMPT,
    _parse_notes_response,
    build_prompt_text,
    generate_speaker_notes,
)


# ---------------------------------------------------------------------------
# build_prompt_text
# ---------------------------------------------------------------------------


class TestBuildPromptText:
    def test_contains_system_prompt(self):
        prompt = build_prompt_text("transcript", "outline")
        assert "expert presentation coach" in prompt

    def test_contains_outline(self):
        prompt = build_prompt_text("transcript", "Slide 1\nSlide 2")
        assert "Slide 1" in prompt
        assert "Slide 2" in prompt

    def test_outline_wrapped_in_xml_tags(self):
        prompt = build_prompt_text("transcript", "My Slide")
        assert "<outline>" in prompt
        assert "</outline>" in prompt

    def test_contains_transcript(self):
        prompt = build_prompt_text("Hello world transcript", "outline")
        assert "Hello world transcript" in prompt

    def test_transcript_wrapped_in_xml_tags(self):
        prompt = build_prompt_text("my text", "outline")
        assert "<transcript>" in prompt
        assert "</transcript>" in prompt

    def test_short_transcript_not_truncated(self):
        short = "short text"
        prompt = build_prompt_text(short, "outline")
        assert "[transcript truncated]" not in prompt
        assert "short text" in prompt

    def test_long_transcript_is_truncated(self):
        long_transcript = "x" * (_MAX_TRANSCRIPT_CHARS + 10_000)
        prompt = build_prompt_text(long_transcript, "outline")
        assert "[transcript truncated]" in prompt

    def test_truncated_transcript_prefix_is_preserved(self):
        long_transcript = "A" * (_MAX_TRANSCRIPT_CHARS + 5_000)
        prompt = build_prompt_text(long_transcript, "outline")
        assert "A" * _MAX_TRANSCRIPT_CHARS in prompt

    def test_exact_limit_transcript_not_truncated(self):
        exact = "B" * _MAX_TRANSCRIPT_CHARS
        prompt = build_prompt_text(exact, "outline")
        assert "[transcript truncated]" not in prompt

    def test_returns_string(self):
        result = build_prompt_text("transcript", "outline")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _parse_notes_response
# ---------------------------------------------------------------------------


SAMPLE_NOTES = [{"number": 1, "title": "Slide 1", "notes": "Some notes"}]
SAMPLE_JSON = '[{"number": 1, "title": "Slide 1", "notes": "Some notes"}]'


class TestParseNotesResponse:
    def test_clean_json_array(self):
        result = _parse_notes_response(SAMPLE_JSON)
        assert result == SAMPLE_NOTES

    def test_json_in_backtick_json_fence(self):
        raw = f"```json\n{SAMPLE_JSON}\n```"
        result = _parse_notes_response(raw)
        assert result == SAMPLE_NOTES

    def test_json_in_plain_backtick_fence(self):
        raw = f"```\n{SAMPLE_JSON}\n```"
        result = _parse_notes_response(raw)
        assert result == SAMPLE_NOTES

    def test_json_embedded_in_surrounding_text(self):
        raw = f"Here are your notes:\n{SAMPLE_JSON}\nEnd of response."
        result = _parse_notes_response(raw)
        assert result == SAMPLE_NOTES

    def test_multiple_slides_parsed_correctly(self):
        raw = '[{"number": 1, "title": "A", "notes": "n1"}, {"number": 2, "title": "B", "notes": "n2"}]'
        result = _parse_notes_response(raw)
        assert len(result) == 2
        assert result[0]["title"] == "A"
        assert result[1]["title"] == "B"

    def test_empty_array_returns_empty_list(self):
        result = _parse_notes_response("[]")
        assert result == []

    def test_json_with_leading_trailing_whitespace(self):
        raw = f"  \n  {SAMPLE_JSON}  \n  "
        result = _parse_notes_response(raw)
        assert result == SAMPLE_NOTES

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_notes_response("This is not JSON at all")

    def test_json_object_not_list_raises_value_error(self):
        with pytest.raises(ValueError):
            _parse_notes_response('{"number": 1, "title": "x", "notes": "y"}')

    def test_malformed_json_in_fence_raises_value_error(self):
        with pytest.raises(ValueError):
            _parse_notes_response("```json\n{not valid json}\n```")

    def test_raw_snippet_included_in_error_message(self):
        bad_input = "clearly not json"
        with pytest.raises(ValueError) as exc_info:
            _parse_notes_response(bad_input)
        assert bad_input in str(exc_info.value)


# ---------------------------------------------------------------------------
# generate_speaker_notes
# ---------------------------------------------------------------------------


def _make_api_mock(response_text: str) -> tuple[MagicMock, MagicMock]:
    """Return (mock Anthropic class, mock client instance) with canned response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_cls = MagicMock(return_value=mock_client)
    return mock_anthropic_cls, mock_client


class TestGenerateSpeakerNotes:
    def test_returns_parsed_notes_list(self):
        mock_cls, _ = _make_api_mock(SAMPLE_JSON)
        with patch("speakernotes_whisper.claude_client.anthropic.Anthropic", mock_cls):
            result = generate_speaker_notes("transcript", "Slide 1", "fake-key")
        assert result == SAMPLE_NOTES

    def test_client_initialised_with_api_key(self):
        mock_cls, _ = _make_api_mock(SAMPLE_JSON)
        with patch("speakernotes_whisper.claude_client.anthropic.Anthropic", mock_cls):
            generate_speaker_notes("transcript", "Slide 1", "my-secret-key")
        mock_cls.assert_called_once_with(api_key="my-secret-key")

    def test_default_model_is_claude_sonnet(self):
        mock_cls, mock_client = _make_api_mock(SAMPLE_JSON)
        with patch("speakernotes_whisper.claude_client.anthropic.Anthropic", mock_cls):
            generate_speaker_notes("transcript", "outline", "key")
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["model"] == "claude-sonnet-4-6"

    def test_custom_model_is_forwarded(self):
        mock_cls, mock_client = _make_api_mock(SAMPLE_JSON)
        with patch("speakernotes_whisper.claude_client.anthropic.Anthropic", mock_cls):
            generate_speaker_notes("transcript", "outline", "key", model="claude-haiku-4-5-20251001")
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_max_tokens_default(self):
        mock_cls, mock_client = _make_api_mock(SAMPLE_JSON)
        with patch("speakernotes_whisper.claude_client.anthropic.Anthropic", mock_cls):
            generate_speaker_notes("transcript", "outline", "key")
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["max_tokens"] == 8192

    def test_custom_max_tokens_forwarded(self):
        mock_cls, mock_client = _make_api_mock(SAMPLE_JSON)
        with patch("speakernotes_whisper.claude_client.anthropic.Anthropic", mock_cls):
            generate_speaker_notes("transcript", "outline", "key", max_tokens=4096)
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["max_tokens"] == 4096

    def test_long_transcript_is_truncated_before_api_call(self):
        long_transcript = "x" * (_MAX_TRANSCRIPT_CHARS + 10_000)
        mock_cls, mock_client = _make_api_mock(SAMPLE_JSON)
        with patch("speakernotes_whisper.claude_client.anthropic.Anthropic", mock_cls):
            generate_speaker_notes(long_transcript, "outline", "key")

        _, kwargs = mock_client.messages.create.call_args
        content_blocks = kwargs["messages"][0]["content"]
        # Second block carries the (potentially truncated) transcript text
        transcript_block_text = content_blocks[1]["text"]
        assert "[transcript truncated]" in transcript_block_text
        assert len(transcript_block_text) <= _MAX_TRANSCRIPT_CHARS + len("\n[transcript truncated]") + 5

    def test_short_transcript_not_truncated(self):
        short = "short transcript"
        mock_cls, mock_client = _make_api_mock(SAMPLE_JSON)
        with patch("speakernotes_whisper.claude_client.anthropic.Anthropic", mock_cls):
            generate_speaker_notes(short, "outline", "key")

        _, kwargs = mock_client.messages.create.call_args
        content_blocks = kwargs["messages"][0]["content"]
        transcript_block_text = content_blocks[1]["text"]
        assert "[transcript truncated]" not in transcript_block_text
        assert short in transcript_block_text

    def test_transcript_block_has_cache_control(self):
        mock_cls, mock_client = _make_api_mock(SAMPLE_JSON)
        with patch("speakernotes_whisper.claude_client.anthropic.Anthropic", mock_cls):
            generate_speaker_notes("transcript", "outline", "key")

        _, kwargs = mock_client.messages.create.call_args
        content_blocks = kwargs["messages"][0]["content"]
        transcript_block = content_blocks[1]
        assert transcript_block.get("cache_control") == {"type": "ephemeral"}

    def test_system_prompt_is_included(self):
        mock_cls, mock_client = _make_api_mock(SAMPLE_JSON)
        with patch("speakernotes_whisper.claude_client.anthropic.Anthropic", mock_cls):
            generate_speaker_notes("transcript", "outline", "key")

        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["system"] == SYSTEM_PROMPT
