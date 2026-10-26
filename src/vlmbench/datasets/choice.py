"""Reading the letter a model chose, for multiple-choice benchmarks we grade ourselves.

The VisualOverload judge (``visualoverload_extract.clean_multiple_choice_answer``) falls back to
the first option letter appearing *anywhere* in a reply, which misreads explanations ("Above
the cart ... so the answer is B" -> "A"). That is right for VisualOverload, where we must match
the official server, but wrong for benchmarks with public answers. This module only accepts a
letter the reply clearly commits to.
"""

from __future__ import annotations

import re

LETTERS = "ABCDEFGHIJK"

_LONE_LETTER = re.compile(r"^[\s*_`(\[]*([A-K])[\s*_`)\].:]*$")
_LETTER_WITH_TEXT = re.compile(r"^[\s*_`(\[]*([A-K])[.)]\s+(.+?)[\s*_`)\].]*$")
_STATED = re.compile(r"(?:answer|option|choice)\s*(?:is|:|would be|=)?\s*[*_`\s]*\(?([A-K])\b(?![a-z'])",
                     flags=re.IGNORECASE)
_MD_WRAP = re.compile(r"^(\*\*|__|\*|_|`+)((?:(?!\1).)+)\1$", flags=re.DOTALL)


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _strip_markdown(s: str) -> str:
    s = s.strip()
    while (m := _MD_WRAP.match(s)) is not None:
        s = m.group(2).strip()
    return s


def extract_choice_letter(response: str | None, options) -> str | None:
    """The letter a reply commits to, or None. Accepted, in order: an <answer> tag, a reply that
    is just a letter, the last line that is just a letter, a final line with a letter followed
    by that option's own text ("A. Front left"), or the last "answer/option is X"."""
    if not response:
        return None
    valid = {LETTERS[i] for i in range(len(options))}
    tagged = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    text = _strip_markdown(tagged[-1] if tagged else response)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for candidate in [text] + lines[::-1]:
        m = _LONE_LETTER.match(candidate)
        if m:
            return m.group(1) if m.group(1) in valid else None
    m = _LETTER_WITH_TEXT.match(lines[-1]) if lines else None  # final line only, not option lists
    if m and m.group(1) in valid:
        said, option = _squash(m.group(2)), _squash(options[LETTERS.index(m.group(1))])
        if said and option and (said.startswith(option) or option.startswith(said)):
            return m.group(1)
    stated = [m.group(1).upper() for m in _STATED.finditer(text)]
    stated = [s for s in stated if s in valid]
    return stated[-1] if stated else None
