"""AI Academic Tutor.

Streamlit app for solving academic problems from text or images through an
OpenAI-compatible vision API (xAI Grok by default).
"""

from __future__ import annotations

import base64
import os
import re
from typing import Any

import streamlit as st
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.x.ai/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "grok-4.7")
DEFAULT_TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are an expert academic tutor for STEM subjects
(mathematics, physics, chemistry) and general academic subjects.

Your goal is to help the student understand the reasoning, not merely provide
an answer to copy.

## Teaching principles
- Explain why a method is chosen and which law, theorem, definition, or
  formula is being used.
- Show the reasoning clearly and do not skip important algebraic steps.
- Match the explanation to the student's apparent level. If unclear, assume
  upper-secondary or early undergraduate level.
- Reply in the language used by the student.
- Prefer accuracy over confidence.
- Never invent values, labels, diagram features, or text that you cannot read.
- If the image is unclear, explicitly identify what is unreadable and ask for
  a clearer image instead of guessing.

## Problem-solving format
For a normal problem, use this structure when useful:

1. **Problem** — Restate the problem and identify the given information.
2. **Key concepts** — Name the relevant concepts, laws, formulas, or theorems.
3. **Step-by-step solution** — Number the steps and explain the purpose of
   each step. Show substitutions and important intermediate algebra.
4. **Final answer** — Give the answer clearly, with units and appropriate
   precision.
5. **Check** — Briefly verify the result using substitution, dimensional
   analysis, an estimate, or another suitable check.

For simple questions, use a shorter answer instead of forcing this structure.

## Images and diagrams
- Carefully inspect all visible text, symbols, equations, tables, graphs, and
  diagrams before solving.
- Distinguish what is clearly visible from what is assumed.
- If an essential part cannot be read, stop and ask the student for a clearer
  image rather than fabricating it.
- If a diagram has multiple labels or parts, preserve their relationships.

## Student's own work
If the student provides an attempted solution:
- Identify the first incorrect step.
- Explain exactly why it is incorrect.
- Preserve and acknowledge the parts that are correct.
- Then show the corrected reasoning.

## Follow-up questions
For a follow-up:
- Address the exact step or concept being asked about.
- Re-explain it from a different angle, with smaller steps or a simple
  example when useful.
- Do not unnecessarily repeat the entire original solution.

## Formatting
- Use Markdown.
- Use LaTeX for mathematics: $...$ for inline expressions and $$...$$ for
  important displayed equations.
- Use upright units, for example $9.8\\ \\text{m/s}^2$.
- Use SI units unless the problem specifies another system.
- Use significant figures for measured quantities, not exact mathematical
  results.

## Boundaries
- If a problem is ill-posed or genuinely cannot be solved from the supplied
  information, say why and identify exactly what is missing.
- If the question is outside academic tutoring, answer briefly and politely
  redirect when appropriate.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_secret(name: str, default: str = "") -> str:
    """Return an environment variable, then a Streamlit secret."""
    value = os.getenv(name)
    if value:
        return value.strip()

    try:
        secret_value = st.secrets.get(name, default)
        return str(secret_value).strip() if secret_value else default
    except Exception:
        # st.secrets can raise when no secrets file/configuration exists.
        return default


def normalize_latex(text: str) -> str:
    """Convert common LaTeX delimiters to Streamlit-friendly delimiters."""
    if not text:
        return ""

    # Display math: \[ ... \] -> $$ ... $$
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

    # Inline math: \( ... \) -> $ ... $
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)

    return text


def build_image_data_url(image_bytes: bytes, mime_type: str | None) -> str:
    """Build a data URL accepted by OpenAI-compatible vision APIs."""
    mime = (mime_type or "image/jpeg").strip().lower()

    # Some browsers may return a generic MIME type. Infer a safe default.
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        mime = "image/jpeg"

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def to_api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert app messages into OpenAI Chat Completions message format."""
    api_messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for message in messages:
        role = message.get("role")
        text = str(message.get("text") or "")
        images = message.get("images") or []

        if role not in {"user", "assistant"}:
            continue

        if images:
            content: list[dict[str, Any]] = []

            if text:
                content.append({"type": "text", "text": text})

            for image_bytes, mime_type in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": build_image_data_url(image_bytes, mime_type),
                            "detail": "high",
                        },
                    }
                )

            api_messages.append({"role": role, "content": content})
        else:
            api_messages.append({"role": role, "content": text})

    return api_messages


def extract_stream_text(chunk: Any) -> str:
    """Safely extract text from an OpenAI-compatible streaming chunk."""
    try:
        if not chunk.choices:
            return ""
        return chunk.choices[0].delta.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def stream_reply(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Stream a model response into the UI and return the complete response."""
    placeholder = st.empty()
    full_text = ""

    stream = client.chat.completions.create(
        model=model,
        messages=to_api_messages(messages),
        stream=True,
        temperature=temperature,
    )

    for chunk in stream:
        delta = extract_stream_text(chunk)
        if not delta:
            continue

        full_text += delta
        placeholder.markdown(normalize_latex(full_text) + "▌")

    placeholder.markdown(normalize_latex(full_text))

    if not full_text.strip():
        raise RuntimeError(
            "The model returned an empty response. Check the selected model "
            "and API provider."
        )

    return full_text


def render_message(message: dict[str, Any]) -> None:
    """Render one stored conversation message."""
    role = message.get("role", "assistant")

    with st.chat_message(role):
        for image_bytes, _mime_type in message.get("images", []) or []:
            st.image(image_bytes, width="stretch")

        text = message.get("text") or ""
        if text:
            st.markdown(normalize_latex(text))


def generate_reply(client: OpenAI, model: str) -> bool:
    """Generate an assistant response for the current conversation.

    Returns True on success and False if the request failed.
    """
    with st.chat_message("assistant"):
        try:
            reply = stream_reply(
                client=client,
                model=model,
                messages=st.session_state.messages,
            )
        except Exception as exc:
            st.error(
                "The request could not be completed. "
                "Check your API key, base URL, model name, and image support."
            )
            with st.expander("Technical details"):
                st.code(str(exc))

            # Remove the unanswered user message so the conversation remains
            # internally consistent.
            if st.session_state.messages:
                last = st.session_state.messages[-1]
                if last.get("role") == "user":
                    st.session_state.messages.pop()

            return False

    st.session_state.messages.append(
        {"role": "assistant", "text": reply, "images": []}
    )
    return True


def reset_conversation() -> None:
    """Start a new problem."""
    st.session_state.messages = []
    st.rerun()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Academic Tutor",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 AI Academic Tutor")
st.caption(
    "Upload a problem photo or type a question and work through the solution "
    "step by step."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "API key",
        type="password",
        value=get_secret("LLM_API_KEY"),
        help=(
            "You can also set LLM_API_KEY as an environment variable "
            "or Streamlit secret."
        ),
    )

    base_url = st.text_input(
        "API base URL",
        value=DEFAULT_BASE_URL,
        help="Example for xAI: https://api.x.ai/v1",
    )

    model = st.text_input(
        "Vision model",
        value=DEFAULT_MODEL,
        help="The selected model must support image input.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
        help="Lower values generally give more deterministic academic answers.",
    )

    st.divider()

    if st.button("🗑️ New problem", use_container_width=True):
        reset_conversation()

if not api_key:
    st.info("Enter your API key in the sidebar to get started.")
    st.stop()

if not base_url.strip():
    st.error("API base URL cannot be empty.")
    st.stop()

if not model.strip():
    st.error("Model name cannot be empty.")
    st.stop()

client = OpenAI(
    api_key=api_key.strip(),
    base_url=base_url.strip(),
)

# ---------------------------------------------------------------------------
# Conversation display
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    render_message(message)


# ---------------------------------------------------------------------------
# First problem
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    uploaded = st.file_uploader(
        "Upload a problem image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Use a clear, well-lit image where all equations and labels are readable.",
    )

    typed = st.text_area(
        "Or type the problem / add context",
        placeholder=(
            "Example: Solve part (b) only.\n"
            "Or: Here is my attempt. Where did I make the first mistake?"
        ),
    )

    if uploaded is not None:
        st.image(uploaded, width="stretch")

    has_input = uploaded is not None or bool(typed.strip())

    if st.button(
        "Solve step by step",
        type="primary",
        disabled=not has_input,
        use_container_width=True,
    ):
        images: list[tuple[bytes, str]] = []

        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            if not image_bytes:
                st.error("The uploaded image is empty. Please choose another image.")
                st.stop()

            images.append(
                (
                    image_bytes,
                    uploaded.type or "image/jpeg",
                )
            )

        text = typed.strip()
        if not text and images:
            text = "Please solve the academic problem shown in this image step by step."

        user_message = {
            "role": "user",
            "text": text,
            "images": images,
        }

        st.session_state.messages.append(user_message)
        render_message(user_message)

        if generate_reply(client, model.strip()):
            st.rerun()


# ---------------------------------------------------------------------------
# Follow-up conversation
# ---------------------------------------------------------------------------

else:
    follow_up = st.chat_input(
        "Ask a follow-up question, request a re-explanation, or check a step..."
    )

    if follow_up and follow_up.strip():
        user_message = {
            "role": "user",
            "text": follow_up.strip(),
            "images": [],
        }

        st.session_state.messages.append(user_message)
        render_message(user_message)

        generate_reply(client, model.strip())
