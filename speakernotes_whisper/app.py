from __future__ import annotations

import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from speakernotes_whisper.claude_client import build_prompt_text, generate_speaker_notes, _parse_notes_response
from speakernotes_whisper.pptx_writer import extract_slide_titles_from_pptx, write_notes_to_pptx
from speakernotes_whisper.transcribe import load_model, transcribe_audio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
CLAUDE_MODELS = ["claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-4-5-20251001"]
SUPPORTED_AUDIO_TYPES = ["mp3", "mp4", "wav", "m4a", "ogg", "webm", "mkv", "mov"]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading Whisper model (first run may download it)...")
def _get_whisper_model(model_size: str):
    return load_model(model_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _save_upload_to_temp(uploaded_file) -> str:
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def _render_notes_as_text(notes: list[dict]) -> str:
    lines: list[str] = []
    for slide in notes:
        lines.append(f"=== Slide {slide['number']}: {slide['title']} ===")
        lines.append(slide["notes"])
        lines.append("")
    return "\n".join(lines)


def _get_api_key() -> str | None:
    return st.session_state.get("api_key") or os.getenv("ANTHROPIC_API_KEY") or None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar():
    st.sidebar.title("Settings")

    st.sidebar.subheader("Mode")
    mode = st.sidebar.radio(
        "How to generate notes",
        ["Claude API", "Copy prompt"],
        index=["Claude API", "Copy prompt"].index(
            st.session_state.get("mode", "Copy prompt")
        ),
        help=(
            "**Claude API**: calls the API automatically (requires an API key).\n\n"
            "**Copy prompt**: assembles the prompt so you can paste it into claude.ai."
        ),
    )
    st.session_state["mode"] = mode

    if mode == "Claude API":
        st.sidebar.subheader("Anthropic API Key")
        default_key = os.getenv("ANTHROPIC_API_KEY", "")
        key_input = st.sidebar.text_input(
            "API Key",
            value=st.session_state.get("api_key", default_key),
            type="password",
            help="Your Anthropic API key. Set ANTHROPIC_API_KEY in .env to avoid entering it each time.",
        )
        st.session_state["api_key"] = key_input

    st.sidebar.subheader("Whisper Model")
    whisper_model = st.sidebar.selectbox(
        "Model size",
        WHISPER_MODELS,
        index=WHISPER_MODELS.index(st.session_state.get("whisper_model", "base")),
        help="Larger models are more accurate but slower. 'base' is a good starting point.",
    )
    st.session_state["whisper_model"] = whisper_model

    st.sidebar.subheader("Claude Model")
    claude_model = st.sidebar.selectbox(
        "Model",
        CLAUDE_MODELS,
        index=CLAUDE_MODELS.index(st.session_state.get("claude_model", CLAUDE_MODELS[0])),
    )
    st.session_state["claude_model"] = claude_model

    # Navigation shortcut — let user jump back to any completed step
    step = st.session_state.get("step", "upload")
    if step != "upload":
        st.sidebar.divider()
        if st.sidebar.button("Start Over"):
            # Clean up temp file if present
            if "audio_path" in st.session_state:
                try:
                    os.unlink(st.session_state["audio_path"])
                except OSError:
                    pass
            st.session_state.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Step renderers
# ---------------------------------------------------------------------------


def _render_step_upload():
    st.header("Step 1 — Upload Your Recording")
    st.write("Upload the audio or video file from your presentation dry-run.")

    if not shutil.which("ffmpeg"):
        st.error(
            "**ffmpeg not found.** faster-whisper requires ffmpeg to process audio/video files.\n\n"
            "Install it with:\n"
            "- macOS: `brew install ffmpeg`\n"
            "- Ubuntu/Debian: `sudo apt install ffmpeg`\n"
            "- Windows: download from https://ffmpeg.org/download.html"
        )
        return

    uploaded = st.file_uploader(
        "Audio or video file",
        type=SUPPORTED_AUDIO_TYPES,
        help="Supports MP3, MP4, WAV, M4A, OGG, WebM, MKV, MOV. Large files may take a moment to upload.",
    )

    if uploaded:
        st.success(f"Loaded: **{uploaded.name}** ({_fmt_bytes(uploaded.size)})")
        if st.button("Continue to Transcription →"):
            audio_path = _save_upload_to_temp(uploaded)
            st.session_state["audio_path"] = audio_path
            st.session_state["audio_filename"] = uploaded.name
            st.session_state["step"] = "transcribe"
            st.rerun()


def _render_step_transcribe():
    st.header("Step 2 — Transcribe Audio")
    st.write(f"File: **{st.session_state.get('audio_filename', '')}**")

    model_size = st.session_state.get("whisper_model", "base")
    st.info(f"Using Whisper model: **{model_size}**. Change in the sidebar if needed.")

    # Show existing transcript if available (user came back to this step)
    if "transcript" in st.session_state:
        st.success("Transcription complete.")
        with st.expander("View transcript"):
            st.text_area(
                "Transcript",
                st.session_state["transcript"],
                height=300,
                disabled=True,
            )
        if st.button("Continue to Slide Outline →"):
            st.session_state["step"] = "outline"
            st.rerun()
        return

    if st.button("Start Transcription"):
        model = _get_whisper_model(model_size)
        progress_bar = st.progress(0, text="Starting transcription...")

        def _on_progress(pct: float, msg: str):
            progress_bar.progress(pct, text=msg)

        with st.spinner("Transcribing..."):
            result = transcribe_audio(
                st.session_state["audio_path"],
                model=model,
                progress_callback=_on_progress,
            )

        progress_bar.progress(1.0, text="Done!")
        st.session_state["transcript"] = result.text
        st.session_state["segments"] = result.segments
        st.session_state["language"] = result.language
        st.session_state["duration"] = result.duration_seconds

        duration_min = int(result.duration_seconds / 60)
        duration_sec = int(result.duration_seconds % 60)
        st.success(
            f"Transcribed **{len(result.segments)} segments** "
            f"({duration_min}m {duration_sec}s, language: {result.language})"
        )

        with st.expander("View transcript"):
            st.text_area("Transcript", result.text, height=300, disabled=True)

        if st.button("Continue to Slide Outline →"):
            st.session_state["step"] = "outline"
            st.rerun()


def _render_step_outline():
    st.header("Step 3 — Slide Outline")
    st.write(
        "Enter your slide titles below, **one per line**. "
        "Or upload your .pptx file to extract titles automatically."
    )

    col_text, col_pptx = st.columns([3, 2])

    with col_pptx:
        st.subheader("Auto-extract from .pptx")
        pptx_file = st.file_uploader(
            "Upload .pptx (optional)",
            type=["pptx"],
            key="pptx_uploader",
            help="Titles will be extracted from each slide's title placeholder.",
        )
        if pptx_file:
            extracted = extract_slide_titles_from_pptx(pptx_file.getbuffer())
            st.session_state["outline_prefill"] = extracted
            st.session_state["pptx_bytes"] = bytes(pptx_file.getbuffer())
            st.success(f"Extracted {len(extracted.splitlines())} slide titles.")

    with col_text:
        st.subheader("Slide titles")
        default_outline = st.session_state.get(
            "outline_prefill",
            st.session_state.get("outline", ""),
        )
        outline_text = st.text_area(
            "One title per line",
            value=default_outline,
            height=320,
            placeholder="Introduction\nProblem Statement\nOur Solution\nDemo\nConclusion",
        )

    if outline_text and outline_text.strip():
        mode = st.session_state.get("mode", "Copy prompt")
        btn_label = "Generate Speaker Notes →" if mode == "Claude API" else "Build Prompt →"
        next_step = "generate" if mode == "Claude API" else "copy_prompt"
        if st.button(btn_label):
            st.session_state["outline"] = outline_text.strip()
            st.session_state["step"] = next_step
            st.rerun()
    else:
        st.warning("Enter at least one slide title to continue.")


def _render_step_generate():
    st.header("Step 4 — Generating Speaker Notes")

    api_key = _get_api_key()
    if not api_key:
        st.error(
            "No Anthropic API key found. "
            "Enter your key in the sidebar or set `ANTHROPIC_API_KEY` in your `.env` file."
        )
        if st.button("← Back to Outline"):
            st.session_state["step"] = "outline"
            st.rerun()
        return

    with st.spinner("Calling Claude API — this usually takes 10–30 seconds..."):
        try:
            notes = generate_speaker_notes(
                transcript=st.session_state["transcript"],
                outline=st.session_state["outline"],
                api_key=api_key,
                model=st.session_state.get("claude_model", "claude-sonnet-4-6"),
            )
        except Exception as exc:
            st.error(f"Error calling Claude API:\n\n{exc}")
            if st.button("← Back to Outline"):
                st.session_state["step"] = "outline"
                st.rerun()
            return

    st.session_state["notes"] = notes
    st.session_state["step"] = "output"
    st.rerun()


def _render_step_copy_prompt():
    st.header("Step 4 — Copy Prompt to your AI")
    st.write(
        "Copy the prompt below and paste it into a new conversation in "
        "the AI tool of your choice. Then paste the response back here."
    )

    prompt_text = build_prompt_text(
        transcript=st.session_state["transcript"],
        outline=st.session_state["outline"],
    )

    st.subheader("Your prompt")
    st.text_area(
        "Select all and copy (Cmd+A / Ctrl+A, then Cmd+C / Ctrl+C)",
        value=prompt_text,
        height=300,
        key="prompt_display",
    )
    st.caption(f"Prompt length: {len(prompt_text):,} characters")

    st.divider()
    st.subheader("Paste Claude's response")
    st.write(
        "After Claude replies, copy its entire response and paste it below. "
        "The app will parse the JSON and take you to the output step."
    )

    pasted = st.text_area(
        "Paste response here",
        height=200,
        key="claude_response_paste",
        placeholder='[{"number": 1, "title": "...", "notes": "..."}, ...]',
    )

    if pasted and pasted.strip():
        if st.button("Parse response & view notes →"):
            try:
                notes = _parse_notes_response(pasted)
            except ValueError as exc:
                st.error(
                    f"Could not parse the response as speaker notes JSON.\n\n{exc}\n\n"
                    "Make sure you copied Claude's full reply and that it contains a JSON array."
                )
            else:
                st.session_state["notes"] = notes
                st.session_state["step"] = "output"
                st.rerun()


def _render_step_output():
    st.header("Step 5 — Your Speaker Notes")

    notes: list[dict] = st.session_state["notes"]

    # Per-slide display (editable so the user can tweak before downloading)
    st.subheader("Review & edit")
    edited_notes: list[dict] = []
    for slide in notes:
        with st.expander(f"Slide {slide['number']}: {slide['title']}", expanded=False):
            edited_text = st.text_area(
                "Notes",
                value=slide["notes"],
                height=150,
                key=f"note_edit_{slide['number']}",
            )
            edited_notes.append({**slide, "notes": edited_text})

    st.divider()
    st.subheader("Download")

    col1, col2 = st.columns(2)

    with col1:
        txt_content = _render_notes_as_text(edited_notes)
        st.download_button(
            label="Download as .txt",
            data=txt_content,
            file_name="speaker_notes.txt",
            mime="text/plain",
        )

    with col2:
        pptx_bytes = st.session_state.get("pptx_bytes")
        if pptx_bytes:
            result_pptx = write_notes_to_pptx(pptx_bytes, edited_notes)
            st.download_button(
                label="Download .pptx with notes",
                data=result_pptx,
                file_name="presentation_with_notes.pptx",
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".presentationml.presentation"
                ),
            )
        else:
            st.info("Upload a .pptx in step 3 to enable direct PowerPoint export.")

    st.divider()
    if st.button("Re-generate (same outline)"):
        st.session_state.pop("notes", None)
        st.session_state["step"] = "generate"
        st.rerun()


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

_STEPS_API = {
    "upload": (1, "Upload"),
    "transcribe": (2, "Transcribe"),
    "outline": (3, "Outline"),
    "generate": (4, "Generate"),
    "output": (5, "Output"),
}
_STEPS_COPY = {
    "upload": (1, "Upload"),
    "transcribe": (2, "Transcribe"),
    "outline": (3, "Outline"),
    "copy_prompt": (4, "Copy Prompt"),
    "output": (5, "Output"),
}


def _get_steps() -> dict:
    mode = st.session_state.get("mode", "Copy prompt")
    return _STEPS_COPY if mode == "Copy prompt" else _STEPS_API


def main():
    st.set_page_config(
        page_title="Slide Whisper",
        page_icon="🎙",
        layout="wide",
    )
    st.title("Slide Whisper")
    st.caption("Generate speaker notes from your presentation dry-run using Whisper + Claude.")

    _render_sidebar()

    step = st.session_state.get("step", "upload")
    steps = _get_steps()

    # Progress indicator
    current_num, _ = steps.get(step, (1, ""))
    st.progress(
        (current_num - 1) / (len(steps) - 1) if current_num > 1 else 0,
        text=" → ".join(
            f"**{label}**" if num == current_num else label
            for label, (num, label) in steps.items()
        ),
    )
    st.divider()

    if step == "upload":
        _render_step_upload()
    elif step == "transcribe":
        _render_step_transcribe()
    elif step == "outline":
        _render_step_outline()
    elif step == "generate":
        _render_step_generate()
    elif step == "copy_prompt":
        _render_step_copy_prompt()
    elif step == "output":
        _render_step_output()


if __name__ == "__main__":
    main()
