"""Answer extraction, ported from the official VisualOverload evaluation server.

Source: https://huggingface.co/spaces/paulgavrikov/visualoverload-submit/blob/main/judge.py
(c) Paul Gavrikov et al., CC BY-SA 4.0. Ported with minimal changes (None-safety and an
explicit "parsed" flag) so local diagnostics match what the server will score. Ground
truth is private, so this module can tell us *what* would be scored, not whether it is right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WORDS_TO_INT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1000000,
}

_REPLACE_CHARS = {
    "ä": "a", "á": "a", "à": "a", "â": "a", "ã": "a", "å": "a", "ā": "a",
    "ö": "o", "ó": "o", "ò": "o", "ô": "o", "õ": "o", "ō": "o",
    "ü": "u", "ú": "u", "ù": "u", "û": "u", "ū": "u",
    "é": "e", "ĕ": "e", "ė": "e", "ę": "e", "ě": "e",
    "ç": "c", "ć": "c", "č": "c", "ñ": "n", "ń": "n", "ł": "l", "ś": "s", "š": "s",
    "ź": "z", "ż": "z", "ý": "y", "ŷ": "y", "ÿ": "y", "œ": "oe", "æ": "ae",
    "v": "u",
}


def _answer_tag(answer: str) -> str:
    m = re.findall(r"<answer>(.*?)</answer>", answer, flags=re.DOTALL)
    return m[0] if m else answer


def option_labels(options) -> list[str]:
    return [f"{chr(65 + i)}. {o.strip()}" for i, o in enumerate(options)]


def clean_multiple_choice_answer(answer: str | None, labels: list[str]) -> str:
    if answer is None:
        return ""
    answer = _answer_tag(answer)
    clean = answer.strip()
    # (sic) the server checks the un-stripped answer for these prefixes/suffixes
    for prefix in ("Answer:", "The answer is ", "The best answer is "):
        if answer.startswith(prefix):
            clean = clean.replace(prefix, "").strip()
    if answer.endswith("."):
        clean = clean[:-1].strip()
    if len(clean) > 1:  # may contain the full label, e.g. "A. option text"
        for label in labels:
            if label in clean:
                clean = label[0]
                break
    if len(clean) > 1:  # may contain the bare letter somewhere
        for label in labels:
            if label[0] in clean:
                clean = label[0]
                break
    return clean


def clean_counting_answer(answer: str | None) -> tuple[int, bool]:
    """Returns (value, parsed). The server scores unparseable answers as 0."""
    if answer is None or answer.strip() == "":
        return 0, False
    answer = _answer_tag(answer)
    clean = answer.strip().lower()
    if clean.endswith("."):
        clean = clean[:-1].strip()
    if clean in WORDS_TO_INT:
        return WORDS_TO_INT[clean], True
    try:
        return int(round(float(clean))), True
    except ValueError:
        pass
    m = re.search(r"\d+(?:\.\d+)?(?=(?![\s\S]*\d))", clean)  # LAST number in the string
    if m:
        return int(round(float(m.group(0)))), True
    for word, number in WORDS_TO_INT.items():  # FIRST spelled-out number (in dict order)
        if word in clean:
            return number, True
    return 0, False


def clean_text(s: str) -> str:
    s = s.strip().lower()
    for char, rep in _REPLACE_CHARS.items():
        s = s.replace(char, rep)
    for ch in ("\t", "\n", ".", "&", ";", ",", "-", "–"):
        s = s.replace(ch, "")
    s = s.replace("’", "'").replace(":", "").replace("·", " ").replace("'", "").replace("“", "") \
        .replace("”", "").replace('"', "").replace("•", " ").replace("  ", " ")
    return s


def extract_text_from_quotes(s: str) -> str:
    pattern = r"'([^']*?)\"|\"([^\"]*?)\"|`([^`]*?)`|“([^”]*?)”|‘([^’]*?)’"
    matches = re.findall(pattern, s)
    if matches:
        return [g for group in matches for g in group if g][0]
    return s


def clean_ocr_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    return clean_text(extract_text_from_quotes(_answer_tag(answer).strip()))


_MD_WRAP = re.compile(r"^(\*\*|__|\*|_|`+)((?:(?!\1).)+)\1$", flags=re.DOTALL)


def normalize_response(response: str | None) -> str | None:
    """Our only pre-submission cleanup (the paper allows any): strip markdown emphasis
    around the whole answer, e.g. "**1557**" -> "1557", which the server's OCR cleaning
    keeps verbatim and would score as wrong."""
    if response is None:
        return None
    s = response.strip()
    while (m := _MD_WRAP.match(s)) is not None:
        s = m.group(2).strip()
    return s


@dataclass
class Extracted:
    value: str | int
    parsed: bool  # False when the server would fall back to an empty / zero answer


def extract(response: str | None, question_type: str, options=()) -> Extracted:
    response = normalize_response(response)
    if question_type == "choice":
        labels = option_labels(options)
        v = clean_multiple_choice_answer(response, labels)
        return Extracted(v, v in {lab[0] for lab in labels})
    if question_type == "counting":
        v, ok = clean_counting_answer(response)
        return Extracted(v, ok)
    if question_type == "ocr":
        v = clean_ocr_answer(response)
        return Extracted(v, bool(v))
    raise ValueError(f"unknown question_type {question_type!r}")
