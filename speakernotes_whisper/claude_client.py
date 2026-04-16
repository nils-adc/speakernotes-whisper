from __future__ import annotations

import json
import re

import anthropic

SYSTEM_PROMPT = """You are an expert presentation coach. Your task is to analyze a spoken \
presentation transcript and generate clear, useful speaker notes for each slide.

Speaker notes should:
- Capture the key points the presenter made for that slide
- Be written as notes the presenter would read (first person where natural)
- Include specific examples, statistics, or stories mentioned in that portion of the talk
- Be concise but complete — 3–8 sentences per slide is ideal
- Reflect the presenter's own words and style, not generic filler

You will receive an ordered slide outline and a full transcript. Intelligently segment the \
transcript across the slides, infer slide transitions from context, and write the notes."""

# ~45k tokens — well above a 30-min transcript (~6k tokens) but guards against edge cases
_MAX_TRANSCRIPT_CHARS = 180_000


def build_prompt_text(transcript: str, outline: str) -> str:
    """
    Assemble the full prompt as a single copyable string for pasting into Claude.ai.
    Combines the system context and user content into one message.
    """
    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:_MAX_TRANSCRIPT_CHARS] + "\n[transcript truncated]"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "---\n\n"
        "Here is my slide outline (slides in presentation order):\n"
        f"<outline>\n{outline}\n</outline>\n\n"
        "Here is the full transcript of my presentation dry-run:\n"
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        "Please generate speaker notes for each slide. "
        "Return your response as a JSON array with this exact structure:\n"
        "[\n"
        '  {"number": 1, "title": "exact slide title from outline", '
        '"notes": "speaker notes for this slide"},\n'
        "  ...\n"
        "]\n\n"
        "Important:\n"
        "- Include every slide from the outline, even if you cannot find relevant content "
        '(write "Content not covered in this recording")\n'
        "- Preserve the slide order from the outline\n"
        "- Return ONLY valid JSON — no markdown fences, no commentary"
    )


def generate_speaker_notes(
    transcript: str,
    outline: str,
    api_key: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 8192,
) -> list[dict]:
    """
    Call the Claude API to map a transcript to per-slide speaker notes.

    Args:
        transcript: Full transcription text from the dry-run recording.
        outline: Newline-separated slide titles in presentation order.
        api_key: Anthropic API key.
        model: Claude model ID.
        max_tokens: Maximum tokens for the response.

    Returns:
        List of dicts: [{"number": 1, "title": "...", "notes": "..."}, ...]
    """
    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:_MAX_TRANSCRIPT_CHARS] + "\n[transcript truncated]"

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is the slide outline (slides in presentation order):\n"
                            f"<outline>\n{outline}\n</outline>\n\n"
                            "Here is the full transcript of the presentation dry-run:\n"
                            "<transcript>\n"
                        ),
                    },
                    {
                        "type": "text",
                        "text": transcript,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "\n</transcript>\n\n"
                            "Please generate speaker notes for each slide. "
                            "Return your response as a JSON array with this exact structure:\n"
                            "[\n"
                            '  {"number": 1, "title": "exact slide title from outline", '
                            '"notes": "speaker notes for this slide"},\n'
                            "  ...\n"
                            "]\n\n"
                            "Important:\n"
                            "- Include every slide from the outline, even if you cannot find "
                            'relevant content (write "Content not covered in this recording")\n'
                            "- Preserve the slide order from the outline\n"
                            "- Return ONLY valid JSON — no markdown fences, no commentary"
                        ),
                    },
                ],
            }
        ],
    )

    raw = response.content[0].text
    return _parse_notes_response(raw)


def _parse_notes_response(raw: str) -> list[dict]:
    """Parse Claude's JSON response, with a regex fallback for markdown-fenced output."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: find the first JSON array in the response
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse Claude response as a JSON array.\n\nRaw response:\n{raw[:500]}"
    )
