# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "accelerate>=1.13.0",
#     "duckdb>=1.4.1",
#     "jedi>=0.17.2,<0.20.0",
#     "marimo",
#     "polars[pyarrow]>=1.34.0",
#     "sqlglot>=27.27.0",
#     "torch>=2.6.0",
#     "transformers>=4.40.0,<5.0",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Marimo and UV

    - marimo is a alternative to Jupyter Notebooks but functions similar overall
    - automatically updating cells based on computaitonal dependencies
    - easy to use visualization and interaction capabilities
    - easy to use app-building
    - lots of integrated features for developing small apps quickly

    To run a marimo app: `marimo run <notebook>`<br>
    To edit a marimo notebook: `marimo edit <notebook>` (we are currently in this view).

    This marimo notebook assumes you have `ollama` running (either as app or via `ollama serve`) to connect to an LLM.
    Usually we will be using huggingface's transformers library instead.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import torch
    from transformers import pipeline as _hf_pipeline

    _pipe_cache = {}

    def chat(model, messages, think=False, max_new_tokens=512):
        """Transformers-based chat wrapper with ollama-compatible return format."""
        if model not in _pipe_cache:
            _pipe_cache[model] = _hf_pipeline(
                "text-generation",
                model=model,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
        result = _pipe_cache[model](messages, max_new_tokens=max_new_tokens)
        content = result[0]["generated_text"][-1]["content"]
        return {"message": {"role": "assistant", "content": content}}

    return (chat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Marimo UI
    """)
    return


@app.cell
def _(mo):
    # a text label that on enter will update the corresponding value
    user_prompt = mo.ui.text(label="Q:", placeholder="Ask the AI...")
    user_prompt  # return to show element
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ollama Chatting
    """)
    return


@app.cell
def _(mo):
    model_sel = mo.ui.dropdown(
        [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        allow_select_none=False,
        label="Model (HuggingFace ID):",
        value="Qwen/Qwen2.5-0.5B-Instruct",
    )
    model_sel
    return (model_sel,)


@app.cell(hide_code=True)
def _(mo):
    submit = mo.ui.run_button()
    submit
    return (submit,)


@app.cell
def _(chat, mo, model_sel, submit):
    mo.stop(not submit.value)
    response = chat(
        model=model_sel.value,
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
        think=False,
    )

    mo.md(response["message"]["content"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forms
    """)
    return


@app.cell
def _(mo):
    slider1 = mo.ui.slider(1, 100)
    slider1
    return (slider1,)


@app.cell
def _(slider1):
    slider1.value
    return


@app.cell
def _(mo):
    form = mo.ui.slider(1, 100).form()
    form
    return


@app.cell
def _(mo):
    def _():
        # Create a form with multiple elements
        form = (
            mo.md("""
            **Multi element form...**

            {name}

            {date}
        """)
            .batch(
                name=mo.ui.text(label="name"),
                date=mo.ui.date(label="date"),
            )
            .form(show_clear_button=True, bordered=False)
        )
        return form


    multi_form = _()
    multi_form
    return (multi_form,)


@app.cell
def _(multi_form):
    multi_form.value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chat UI
    `mo.ui.chat`: A simple conversation interface.
    """)
    return


@app.cell
def _(chat, mo, model_sel):
    # Example usage of a chat.
    def _():  # note: we make this anonymous function to be able to independently set up some stuff
        # Example of a simple conversation.
        SYSTEM_PROMPT = "You are a helpful assistant that answers concisely."

        def simple_conversation(messages, config):  # the custom-"model"
            print("messages:", messages)
            full_msg = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                {"role": m.role, "content": m.content} for m in messages
            ]
            response = chat(
                model=model_sel.value,
                messages=full_msg,
                think=False,
            )

            return response["message"]["content"]

        return simple_conversation


    # here we set up the chat, note that the parameter is the returned simple_conversation function
    simple_con = mo.ui.chat(_())
    simple_con  # return to show element
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Structure of a Prompt

    Technical:
    - A prompt for `pipeline` is either: `str` or `list[dict]`
      - if `str`: then it is the raw string the model is supposed to _complete_
      - if `list[dict]`: then it is a list of the _utterances_ of a conversation

    Examples (for `gemma`-model):
    ```xml
    <start_of_turn>model
    Hi, how can I help you?
    <end_of_turn>
    <start_of_turn>user
    Help me write a good exercise for today's lecture.
    <end_of_turn>
    ```

    ```json
    [
      {"role":"model", "content":"Hi, how can I help you?"},
      {"role":"user", "content":"Help me write a good exercise for today's lecture."},
    ]
    ```

    ## TICOS-E
    ```md
    You are <role>.
    Your job: <Task>.
    Inputs: <what the model will receive and how it’s delimited>.
    Constraints: <time, length, domain assumptions, safety>.
    Output: <exact format/schema + example>.
    Style: <audience, tone, register, citations, language>.
    Evaluation: <self-checklist or rubric to verify before finalizing>.
    ```

    ### + Planning
    ```md
    First, draft a step-by-step plan.
    Then, execute the plan.
    Keep the plan concise, and each step short.
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case Study: Qwen 3's Chat Template
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```toml
    [SETUP]
        [if tools]
            emit system header
            emit system message (if present)
            emit tools block (serialized)
            emit tool-call format instructions
        [else]
            emit system header + message (if present)

    [PRE-PASS (find last real user query)]
        walk messages in reverse
        mark the index of the last user message that is NOT a tool response


    [MAIN LOOP over messages]
        [user / non-first system]
            emit role header + content
        [assistant]
            separate reasoning from content (via field or <think> tags)
            [if message is after the last real user query]
                emit thinking block + content  (always on last, conditionally otherwise)
            [else]
                emit content only
            [if tool calls present]
                serialize and emit each tool call
        [tool]
            open user turn (only on first consecutive tool message)
            emit tool response block
            close user turn (only on last consecutive tool message)

    [FOOTER]
        [if generation prompt requested]
            emit assistant header
            [if thinking disabled]
                emit empty think block
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case Study: Fabric
    > GitHub: [danielmiessler/fabric](https://github.com/danielmiessler/fabric)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - tool for POSIX-style chaining of prompts
    - initially written in `python`, then rewritten in `go`

    /// admonition | General Idea
    The overarching ideas is that interactions with the model follow repeatable "patterns".
    These patterns are then parametrized in a way that allows for easier reuse and chaining.
    ///

    ![](https://github.com/danielmiessler/Fabric/blob/main/docs/images/fabric-summarize.png?raw=true)
    ```bash
    curl -s https://r.jina.ai/peps.python.org/pep-0020/ | fabric -p summarize -s
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case Study: Pi (Agent Framework)

    > GitHub: [badlogic/pi-mono](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - tool for Claude-code style agent work
    - written in `typescript`
    - by default _very unsafe_ but also **very light** on tokens
    - easily extendeable with skills that allow for complicated behaviours like:
        - planning
        - sub-agents
        - code-execution
        - safeguarding
    - extensions can be: skills, prompts, themes, templates

    /// admonition | General Idea
    Meant as a minimal terminal coding harness.
    The tool should conform to your workflows – not the other way around. <br>
    Shareable user packages via npm and git.
    ///

    ![](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/images/interactive-mode.png?raw=true)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chaining of Prompts
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Let's load some data from Wikipedia
    """)
    return


@app.cell
def _(mo):
    wiki_entries = mo.sql(
        f"""
        FROM 'hf://datasets/wikimedia/wikipedia/20231101.en/*.parquet'
        SELECT * limit 10;
        """
    )
    return (wiki_entries,)


@app.cell
def _():
    from utils import compose

    return (compose,)


@app.cell
def _(compose):
    # Example usage of a helper function we are going to make use of.
    plus3 = lambda x: x + 3
    sqr = lambda x: x**2
    print("plus3 → sqr:", compose(plus3, sqr)(3))

    add = lambda x, y: x + y
    print("add → plus3 → sqr:", compose(add, plus3, sqr)(2, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining and trying out Patterns.
    """)
    return


@app.cell
def _(chat, model_sel):
    def summarize(inp, history):
        system_prompt = {
            "role": "system",
            "content": "You are an expert in analyzing texts. Extract the most important points from the given text.",
        }
        user_message = {"role": "user", "content": f"# INPUT:\n{inp}"}
        result = chat(
            model=model_sel.value,
            messages=[*history, system_prompt, user_message],
            think=False,
        )
        answer = result["message"]
        return answer["content"], history + [answer]

    return (summarize,)


@app.cell(disabled=True)
def _(summarize, wiki_entries):
    summarize(wiki_entries[0, "text"][0:512], [])
    return


@app.cell
def _(chat, model_sel):
    def plan(inp, history):
        system_prompt = {
            "role": "system",
            "content": """
            You are a helpful AI Assistant. 
            You are given the users task in the section INPUT.
            First, draft a step-by-step plan, that another person can use to fulfill the task.
            Keep the plan concise, and each step short.
            """,
        }
        user_message = {"role": "user", "content": f"# INPUT:\n{inp}"}
        result = chat(
            model=model_sel.value, messages=[*history, system_prompt, user_message]
        )
        answer = result["message"]
        return answer["content"], history + [answer]

    return (plan,)


@app.cell(disabled=True)
def _(plan):
    plan("Help me plan a birthday party.", [])
    return


@app.cell(disabled=True)
def _(chat, model_sel):
    def questions_in_plan(inp, history):
        system_prompt = {
            "role": "system",
            "content": """
            You are a helpful AI Assistant. 
            You are given the a step-by-step plan in the section INPUT.
            Think through each step, and make a bullet point list of still open questions that need to be answered before the plan is complete.
            """,
        }
        user_message = {"role": "user", "content": f"# INPUT:\n{inp}"}
        result = chat(
            model=model_sel.value, messages=[*history, system_prompt, user_message]
        )
        answer = result["message"]
        return answer["content"], history + [answer]

    return (questions_in_plan,)


@app.cell(disabled=True)
def _(mo, plan, questions_in_plan):
    def _():
        print("Planning...")
        birthday_plan, _ = plan("Help me plan a birthday party.", [])
        print("Extracting...")
        questions, _ = questions_in_plan(
            birthday_plan, []
        )  # we are forgetting the context of the conversation
        return questions


    birthday_result = _()
    mo.md(birthday_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Your Tasks at Home:

    Prompt Engineering (6pts):
    - engineer a prompt that when given text, will **tag** entities with xml-entities
      - `<names>...</names>` for names in the text
      - `<location>...</location>` for named locations in the text
      - apply the pattern onto a wiki text, choosable via dropdown, and output the tagged texts
    - engineer prompt(s) using **TICOS-E**, and apply it to the wikipedia texts that makes the model output structured json with the following key-value pairs:
      - title: the actual title as given in the title column
      - name: the name of the wikipedia entry as inferred from the entry itself
      - summary: a no more than 75 word summary of the contents of the entry
      - keyideas: a _bulletpoint list_ of the three main takeaways from the entry
    - make the model repeat a piece of text, but have it transform all the text within xml-fences for `<leet>` into leet-speak, and have it translate all the text to english which is in `<german>` fences, apply this to the `exercise2_translation.txt`
      - either make the model itself transform the text or extract the contents and apply the model to each instance separately

    Marimo Apps (4pts):
    - make an app with `marimo` in a file `app.py` that loads a `Qwen3.5` model (we recommend 4B or bigger), and allows the user to have a conversation with the following criteria:
      - the conversation starts with the user asking for help with _something_
      - the model will then come up with a plan to make this happen
      - the model will then continually ask for clarification about open questions form the plan, until the user has answered all the open questions, or till the user requests the model to finish
      - the model will then provide one final answer trying to fulfill the users initial task
      - if the model thinks it is done it should say so
    - use the inbuilt Chat UI element to achieve this
    - make use of the Grid View to set up the UI
    - we will run your notebook with the `marimo run --sandbox app.py` command, so please make sure that works
    - hand-in both: the app.py file and the layouts folder as a combined ZIP file
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 1: Named Entity Tagging

    Engineer a prompt that tags person names with `<names>…</names>` and
    named locations with `<location>…</location>`.
    Select a Wikipedia entry from the dropdown and click **Tag Entities**.
    """)
    return


@app.cell
def _(mo, wiki_entries):
    ner_dropdown = mo.ui.dropdown(
        wiki_entries["title"].to_list(),
        allow_select_none=False,
        label="Wikipedia entry:",
        value=wiki_entries[0, "title"],
    )
    ner_dropdown
    return (ner_dropdown,)


@app.cell
def _(mo):
    ner_run = mo.ui.run_button(label="Tag Entities")
    ner_run
    return (ner_run,)


@app.cell
def _(chat, mo, model_sel, ner_dropdown, ner_run, wiki_entries):
    mo.stop(not ner_run.value)
    _idx = wiki_entries["title"].to_list().index(ner_dropdown.value)
    _text = wiki_entries[_idx, "text"][:1500]

    _system = {
        "role": "system",
        "content": (
            "You are an expert named entity recognition system.\n"
            "Task: Tag ALL person names with <names>…</names> and ALL named locations "
            "with <location>…</location>.\n"
            "Inputs: A plain text passage provided by the user.\n"
            "Constraints: Tag every occurrence; do NOT alter any other text.\n"
            "Output: The COMPLETE original text with only the XML entity tags added — "
            "nothing else before or after.\n"
            "Style: Preserve all punctuation, spacing, and capitalisation.\n"
            "Evaluation: Every person name and every named location must be tagged; "
            "no extra words or explanations added."
        ),
    }
    _user = {
        "role": "user",
        "content": f"Please tag the named entities in this text:\n\n{_text}",
    }
    _result = chat(model=model_sel.value, messages=[_system, _user], think=False)
    mo.md(_result["message"]["content"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 2: Structured JSON Extraction (TICOS-E)

    Use the **same entry** selected in Exercise 1.
    Click **Extract JSON** to get a structured JSON with `title`, `name`,
    `summary` (≤ 75 words) and `keyideas` (3 bullet strings).
    """)
    return


@app.cell
def _(mo):
    json_run = mo.ui.run_button(label="Extract JSON")
    json_run
    return (json_run,)


@app.cell
def _(chat, json_run, mo, model_sel, ner_dropdown, wiki_entries):
    import json as _json
    from utils import extract_fenced_content as _efcc

    mo.stop(not json_run.value)
    _idx2 = wiki_entries["title"].to_list().index(ner_dropdown.value)
    _title = wiki_entries[_idx2, "title"]
    _text2 = wiki_entries[_idx2, "text"][:2000]

    _TICOS_SYSTEM = (
        "You are an expert text analyst specialising in Wikipedia content.\n"
        "Task: Extract structured information from a Wikipedia article.\n"
        "Inputs: An article provided with <title> and <article> XML tags.\n"
        "Constraints: summary must be ≤ 75 words; keyideas must contain exactly 3 items.\n"
        "Output: A single valid JSON object — NO markdown code fences — with exactly "
        "these keys:\n"
        '  "title": the exact title from the <title> tag\n'
        '  "name": the Wikipedia entry name as inferred from the article body\n'
        '  "summary": a concise ≤ 75-word summary\n'
        '  "keyideas": a list of exactly 3 key-takeaway strings\n'
        "Style: Formal, encyclopedic, third-person prose.\n"
        "Evaluation: Verify (1) JSON is valid, (2) summary ≤ 75 words, "
        "(3) exactly 3 keyideas."
    )
    _user2 = {
        "role": "user",
        "content": f"<title>{_title}</title>\n<article>{_text2}</article>",
    }
    _result2 = chat(
        model=model_sel.value,
        messages=[{"role": "system", "content": _TICOS_SYSTEM}, _user2],
        think=False,
    )
    _raw = _result2["message"]["content"]
    _json_str = _efcc(_raw)
    try:
        _parsed = _json.loads(_json_str)
        _display_text = f"```json\n{_json.dumps(_parsed, indent=2)}\n```"
    except _json.JSONDecodeError:
        _display_text = _raw
    mo.md(_display_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 3: Leet-Speak & German Translation

    Process `exercise2_translation.txt`:

    - Replace each `<leet>…</leet>` block with its leet-speak version.
    - Replace each `<german>…</german>` block with an English translation.

    Click **Transform Text** to run.
    """)
    return


@app.cell
def _(mo):
    transform_run = mo.ui.run_button(label="Transform Text")
    transform_run
    return (transform_run,)


@app.cell
def _(chat, mo, model_sel, transform_run):
    import re as _re

    mo.stop(not transform_run.value)

    with open("exercise2_translation.txt", encoding="utf-8") as _f:
        _original = _f.read()

    def _leet_convert(fragment):
        r = chat(
            model=model_sel.value,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the given text to leet speak (1337 style). "
                        "Apply these substitutions: a→4, e→3, i→1, o→0, t→7, s→5. "
                        "Output ONLY the converted text — no explanations."
                    ),
                },
                {"role": "user", "content": fragment},
            ],
            think=False,
        )
        return r["message"]["content"].strip()

    def _translate_de(fragment):
        r = chat(
            model=model_sel.value,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the given German text to English. "
                        "Output ONLY the English translation — no explanations."
                    ),
                },
                {"role": "user", "content": fragment},
            ],
            think=False,
        )
        return r["message"]["content"].strip()

    _result3 = _re.sub(
        r"<leet>(.*?)</leet>",
        lambda m: _leet_convert(m.group(1)),
        _original,
        flags=_re.DOTALL,
    )
    _result3 = _re.sub(
        r"<german>(.*?)</german>",
        lambda m: _translate_de(m.group(1)),
        _result3,
        flags=_re.DOTALL,
    )

    mo.md(f"### Transformed Text\n\n{_result3}")
    return


if __name__ == "__main__":
    app.run()
