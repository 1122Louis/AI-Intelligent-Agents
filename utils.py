import inspect
import textwrap
import re
from typing import Callable
from functools import reduce

def compose(*funcs):
    """Helper function to allow chaining of python functions."""
    return reduce(lambda f, g: lambda *x: g(f(*x)), funcs)

def get_tool_description(f):
    name = f.__name__
    desc = inspect.cleandoc(f.__doc__ or "")
    sig = inspect.signature(f)
    return name, desc, str(sig)


def extract_xml_element(text: str, el: str, all_occurences=False):
    """Returns all `el`-xml elements as a list of their contents."""
    mth = re.findall if all_occurences else re.search
    results = mth(rf"<{el}>\s*(.*?)\s*</{el}>", text, re.DOTALL)
    if not all_occurences and results is not None:
        results = results.group(1)
    return results


def ensure_el(text: str, el: str):
    if f"<{el}>" not in text:
        text = f"<{el}>" + text
    if f"</{el}>" not in text:
        text = text + f"</{el}>"
    return text


def extract_fenced_content(text: str):
    """
    Returns the content inside the first Markdown fenced code block,
    or the text itself if no fenced block exists.
    """
    match = re.match(r"```(?:\w+)?\s*([\s\S]*?)\s*```", text, re.MULTILINE)
    return match.group(1).strip() if match else text.strip()


def check_if_tool(tools: list[Callable], query: str):
    try:
        ind = [f.__name__ for f in tools].index(query)
        return tools[ind]
    except ValueError:
        return None


def format_tool_schemas(tools: list[Callable]) -> str:
    schema = """"""
    for func in tools:
        name, desc, args = get_tool_description(func)
        schema += f"""- **{name}**\n"""
        schema += textwrap.indent(f"""{desc}\n""", "  ")
    return schema


def block_print(text: str, *args, width: int = 60, keep_whitespace=True):
    """Print text formatted to a given column width inside a double-line frame."""
    # Define box-drawing characters (double-line)
    tl, tr, bl, br = "╔", "╗", "╚", "╝"
    h, v = "═", "║"

    for arg in args:
        text += str(arg)

    if keep_whitespace:
        paragraphs = text.splitlines() or [""]
        wrapped_lines = []
        for para in paragraphs:
            if para.strip():
                wrapped_lines.extend(textwrap.wrap(para, width=width - 4))
            else:
                wrapped_lines.append("")
    else:
        wrapped_lines = textwrap.wrap(
            text,
            width=width - 4,
            replace_whitespace=True,
            drop_whitespace=False,
        )

    print(f"{tl}{h * (width-2)}{tr}")
    for line in wrapped_lines:
        print(f"{v} {line.ljust(width-4)} {v}")
    print(f"{bl}{h * (width-2)}{br}")


if __name__ == "__main__":
    block_print("Wow, what a cow!\n\nAnd that dog!" * 20)
    block_print("Wow, what a cow!\n\n", "And that dog!" * 25, keep_whitespace=True)
    block_print("Wow, what a cow!\n", "\nAnd that dog!", keep_whitespace=True)
# type: ignore
