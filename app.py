# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "accelerate>=1.13.0",
#     "marimo>=0.23.1",
#     "torch>=2.6.0",
#     "transformers>=4.40.0,<5.0",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full", layout_file="layouts/app.grid.json")


@app.cell
def _(): 
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import torch
    from transformers import pipeline as _hf_pipeline

    # MODEL_ID can be changed to any Qwen instruction-tuned model on HuggingFace.
    # The README recommends 4 B parameters or larger for best quality.
    MODEL_ID = "C:/models/qwen-0.5b"  # local path – no symlinks needed on Windows

    _pipe_cache: dict = {}

    def llm_chat(messages: list, model: str = MODEL_ID, max_new_tokens: int = 1024) -> str:
        """Call a HuggingFace text-generation pipeline and return the assistant reply."""
        if model not in _pipe_cache:
            _pipe_cache[model] = _hf_pipeline(
                "text-generation",
                model=model,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
        result = _pipe_cache[model](messages, max_new_tokens=max_new_tokens)
        return result[0]["generated_text"][-1]["content"]

    return (llm_chat,)


@app.cell
def _(mo, llm_chat):
    # ------------------------------------------------------------------ #
    # System prompts for each conversation phase                          #
    # ------------------------------------------------------------------ #
    _SYSTEM_INITIAL = (
        "You are a helpful planning assistant.\n"
        "The user will ask for help with something. Your response MUST:\n\n"
        "1. Briefly acknowledge the request.\n"
        "2. Present a concise numbered STEP-BY-STEP PLAN.\n"
        "3. List ALL open questions needed to personalise the plan "
        "(number them clearly under a heading '## Open Questions').\n"
        "4. Ask ONLY the FIRST open question — do not ask all at once.\n\n"
        "End your response with this marker on its own line:\n"
        "[PHASE:QUESTIONING]"
    )

    _SYSTEM_QUESTIONING = (
        "You are a helpful planning assistant in the QUESTIONING phase.\n"
        "The conversation history shows the original plan and the "
        "questions already asked.\n\n"
        "Your task:\n"
        "- Briefly acknowledge the user's most recent answer.\n"
        "- If the user says 'done', 'stop', 'finish', 'skip', "
        "or 'no more questions': jump straight to the FINAL phase.\n"
        "- Otherwise check whether any open questions from your plan "
        "are still unanswered.\n"
        "  - If YES: ask the NEXT unanswered question, then end with:\n"
        "    [PHASE:QUESTIONING]\n"
        "  - If NO: write a comprehensive final answer that incorporates "
        "all information gathered, then end with:\n"
        "    [PHASE:DONE]"
    )

    # ------------------------------------------------------------------ #
    # Conversation handler                                                 #
    # ------------------------------------------------------------------ #
    def planner(messages, config):
        # Find the last assistant message to detect the current phase.
        last_asst: str | None = None
        for m in reversed(list(messages)):
            if m.role == "assistant":
                last_asst = m.content
                break

        if last_asst is None:
            # First user message → initial planning phase
            system = _SYSTEM_INITIAL
        elif "[PHASE:DONE]" in last_asst:
            return (
                "I have already delivered the final answer above. "
                "Feel free to start a new conversation whenever you like!"
            )
        elif "[PHASE:QUESTIONING]" in last_asst:
            system = _SYSTEM_QUESTIONING
        else:
            # Fallback: treat as initial if markers are absent
            system = _SYSTEM_INITIAL

        full_msgs = [{"role": "system", "content": system}] + [
            {"role": m.role, "content": m.content} for m in messages
        ]
        return llm_chat(full_msgs)

    # ------------------------------------------------------------------ #
    # Chat UI element                                                       #
    # ------------------------------------------------------------------ #
    chat_ui = mo.ui.chat(
        planner,
        prompts=[
            "Help me plan a birthday party",
            "Help me write a cover letter",
            "Help me organise a trip to Japan",
            "Help me learn to cook Italian food",
            "Help me start a vegetable garden",
        ],
    )
    return (chat_ui,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Planning Assistant

        Ask me for help with **anything**.
        Here is what will happen:

        1. I will build a **step-by-step plan** for your request.
        2. I will ask **clarifying questions** one at a time to personalise the plan.
        3. Once all questions are answered, I will deliver a **tailored final answer**.

        ---

        > **Tip:** type **"done"** or **"no more questions"** at any time
        > to skip the remaining questions and receive the final answer immediately.
        """
    )
    return


@app.cell
def _(chat_ui):
    chat_ui
    return


if __name__ == "__main__":
    app.run()


if __name__ == "__main__":
    app.run()
