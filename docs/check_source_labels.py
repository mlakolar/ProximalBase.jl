#!/usr/bin/env python3
"""Check that comments in src/ cite no numerical rule labels.

Comments in the sources used to name rules of an older, labeled numerical
contract by short labels such as R1, T3, M1, E4 or D2. The rendered manual
(docs/src/numerics.md) defines none of them, so a reader of such a comment has
no page to look the rule up in. A comment states the behavior in words instead
and, where docs/src/numerics.md states that behavior, refers to its section.

This script scans src/**/*.jl and reports each token of the form
<letters><digits> for the label families NR, FT, R, T, M, E, O, A and D that
occurs in a comment. It reads the text of comments, not of code: a Julia scanner
separates them from code, so a variable such as E2 in `E1, E2 = f(x)`, a label
inside a string or a docstring, or a `#` inside a string, a character literal or
a command is not a comment. The scanner handles line comments, nested `#= ... =#`
blocks, strings with escapes and `$(...)` interpolation, triple-quoted strings,
command literals, and tells a character literal such as '#' from the adjoint
operator in `A'`.

CPU names such as "Apple M3" are exempt (see EXEMPT_PREFIXES).

Usage: python3 docs/check_source_labels.py [--root DIR] [--self-test]
Exit status is 1 if a label is found (or a self-test fails), 0 otherwise.
"""
from pathlib import Path
import argparse
import re
import sys
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parent.parent

# A label: a family prefix followed by digits, as a whole word.
LABEL = re.compile(r"\b(?:NR|FT|[RTMEOAD])[0-9]+\b")

# CPU names are not labels: a token of one of the given families directly
# preceded by the given prefix, as "M3" in "Apple M3". Pairs (prefix, family).
EXEMPT_PREFIXES = (("Apple ", "M"),)

MESSAGE = ("{path}:{line}: label {token} in a comment; "
           "state the rule in words and cite the numerics.md section instead")


# Characters that Julia allows after the first one in an identifier, besides
# letters and digits: primes, and combining marks and other symbols above ASCII.
_PRIMES = "\u2032\u2033\u2034\u2057"
_IDENTIFIER_CATEGORIES = {"Lu", "Ll", "Lt", "Lm", "Lo", "Nl", "Nd", "No",
                          "Mn", "Mc", "Me", "Pc", "Sc", "So", "Sk"}


def _is_identifier_char(ch):
    """Whether `ch` can end a Julia identifier (so that a quote after it is a
    suffix, as in x' or f"...", and not the start of a literal)."""
    if not ch:
        return False
    if ch.isascii():
        return ch.isalnum() or ch in "_!"
    return ch in _PRIMES or unicodedata.category(ch) in _IDENTIFIER_CATEGORIES


def _is_adjoint_context(prev):
    """Whether a `'` after the character `prev` is the adjoint operator.
    At the start of the text (prev empty) it opens a character literal."""
    return _is_identifier_char(prev) or (bool(prev) and prev in ")]}'")


def _char_literal_end(text, i):
    """Index just past the character literal that opens at text[i], or None."""
    n = len(text)
    if i + 1 >= n:
        return None
    c = text[i + 1]
    if c == "\\":
        # an escape: the backslash and the character after it, then the rest
        # of the escape (é, \x41, octal, \U0001F600) up to the closing quote
        j = i + 3
        while j < n and j - i <= 14 and text[j] not in "'\n":
            j += 1
        if j < n and text[j] == "'" and i + 2 < n and text[i + 2] != "\n":
            return j + 1
        return None
    if c in "'\n":
        return None
    if i + 2 < n and text[i + 2] == "'":
        return i + 3
    return None


def comment_texts(text):
    """Map line number (1-based) to the comment text on that line.

    A line is present when part of it, or all of it, lies in a comment.
    """
    out = {}
    n = len(text)
    i = 0
    line = 1
    # Frames: ["code", None] for top-level code, ["code", depth] inside $( ... ),
    # ["str", quote_length, interpolating] for strings and ["cmd", quote_length,
    # interpolating] for commands. The scanner looks at the top frame.
    stack = [["code", None]]

    def add(ln, s):
        out.setdefault(ln, []).append(s)

    while i < n:
        c = text[i]
        top = stack[-1]
        if top[0] == "code":
            if c == "\n":
                line += 1
                i += 1
            elif c == "#":
                if text.startswith("#=", i):
                    # a nested block comment, which may span lines
                    depth = 0
                    buf = []
                    while i < n:
                        if text.startswith("#=", i):
                            depth += 1
                            buf.append("#=")
                            i += 2
                        elif text.startswith("=#", i):
                            depth -= 1
                            buf.append("=#")
                            i += 2
                            if depth == 0:
                                break
                        elif text[i] == "\n":
                            add(line, "".join(buf))
                            buf = []
                            line += 1
                            i += 1
                        else:
                            buf.append(text[i])
                            i += 1
                    add(line, "".join(buf))
                else:
                    j = text.find("\n", i)
                    if j < 0:
                        j = n
                    add(line, text[i:j])
                    i = j
            elif c == '"':
                q = 3 if text.startswith('"""', i) else 1
                # a string macro (r"...", raw"...", b"...") does not interpolate
                interp = not (i > 0 and _is_identifier_char(text[i - 1]))
                stack.append(["str", q, interp])
                i += q
            elif c == "`":
                q = 3 if text.startswith("```", i) else 1
                # a command macro (custom`...`) does not interpolate either
                interp = not (i > 0 and _is_identifier_char(text[i - 1]))
                stack.append(["cmd", q, interp])
                i += q
            elif c == "'":
                prev = text[i - 1] if i > 0 else ""
                end = None if _is_adjoint_context(prev) else _char_literal_end(text, i)
                if end is None:
                    i += 1
                else:
                    line += text.count("\n", i, end)
                    i = end
            else:
                if top[1] is not None:
                    if c == "(":
                        top[1] += 1
                    elif c == ")":
                        top[1] -= 1
                        if top[1] == 0:
                            stack.pop()
                i += 1
        else:
            kind = top[0]
            q = top[1]
            quote = '"' if kind == "str" else "`"
            interp = top[2]
            if c == "\\":
                if i + 1 < n and text[i + 1] == "\n":
                    line += 1
                i += 2
            elif c == quote and text.startswith(quote * q, i):
                stack.pop()
                i += q
            elif c == "$" and interp and text.startswith("$(", i):
                stack.append(["code", 1])
                i += 2
            else:
                if c == "\n":
                    line += 1
                i += 1
    return {ln: "".join(parts) for ln, parts in out.items()}


def find_labels(text):
    """List of (line, token) for each label that occurs in a comment of text."""
    hits = []
    for ln, comment in sorted(comment_texts(text).items()):
        for m in LABEL.finditer(comment):
            before = comment[:m.start()]
            token = m.group(0)
            if any(before.endswith(p) and token.startswith(f) for p, f in EXEMPT_PREFIXES):
                continue
            hits.append((ln, m.group(0)))
    return hits


def check_tree(root):
    """Scan root/src/**/*.jl; return (files scanned, list of message lines)."""
    root = Path(root)
    files = sorted((root / "src").rglob("*.jl"))
    messages = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        for ln, token in find_labels(text):
            messages.append(MESSAGE.format(path=rel, line=ln, token=token))
    return len(files), messages


def run(root):
    nfiles, messages = check_tree(root)
    for m in messages:
        print(m)
    print(f"check_source_labels: {nfiles} files scanned, {len(messages)} hits")
    return 1 if messages else 0


# Each case: (name, Julia source, expected [(line, token), ...]).
SELF_TEST_CASES = [
    # flagged
    ("label in a line comment", "x = 1\n# R1 here\ny = 2\n", [(2, "R1")]),
    ("trailing comment after code", "x = f(y)  # see T3\n", [(1, "T3")]),
    ("single-line block comment", "x = 1 #= M1 =# + 2\n", [(1, "M1")]),
    ("multi-line block comment, label on an inner line",
     "a = 1\n#= first\nsecond E4\nthird\n=#\nb = 2\n", [(3, "E4")]),
    ("nested block comment",
     "#= outer #= inner D2 =# still outer =#\ncode\n", [(1, "D2")]),
    ("label on the line after a nested block closes",
     "#= a\n#= b =#\nR3 =#\nx = 1\n", [(3, "R3")]),
    ("two labels on one line", "# (T3, R3)\n", [(1, "T3"), (1, "R3")]),
    ("label after a string that contains a #",
     's = "a # b"  # R1 here\n', [(1, "R1")]),
    ("label after a docstring", '"""\ndoc # R1\n"""\nf() = 1  # E2\n', [(4, "E2")]),
    ("label after an escaped quote in a string",
     's = "a\\" # R1"  # NR1\n', [(1, "NR1")]),
    ("label after a character literal of a quote", "c = '\"'  # FT2\n", [(1, "FT2")]),
    ("label after an escaped character literal", "c = '\\''  # O7\n", [(1, "O7")]),
    ("label after an adjoint", "x = A'  # A1\n", [(1, "A1")]),
    ("label after a backtick command that contains a #",
     "run(`echo # not a comment`)  # D1\n", [(1, "D1")]),
    ("label after a string macro with an escaped backslash",
     'r = r"a\\\\"  # R2\n', [(1, "R2")]),
    ("label after a string with an interpolated string",
     's = "$(f("# x")) #"  # T1\n', [(1, "T1")]),
    ("label after an interpolation in a triple-quoted string",
     'doc = """\n$(f(")")) # R1\n"""  # E3\n', [(3, "E3")]),
    ("label after a quote character literal at the start of the text", "'\"' # R1\n", [(1, "R1")]),
    ("label after an adjoint of a name with a prime", "x\u2032'#' R1\n", [(1, "R1")]),
    ("label after an adjoint of a name with a combining mark", "a\u0302'#' R1\n", [(1, "R1")]),
    ("label in an interpolation of a command",
     "c = `$(# R1\n)`\n", [(1, "R1")]),
    ("label after a command macro literal that looks like an interpolation",
     "c = custom`$(# R1)` # R2\n", [(1, "R2")]),
    ("label after a multi-line command macro literal",
     "c = custom`$(# R1\n)` # R2\n", [(2, "R2")]),
    ("labels next to the CPU name exemption",
     "# Apple R1 and Apple D2 and Apple M3\n", [(1, "R1"), (1, "D2")]),
    # not flagged
    ("code after a character literal # at the start of the text", "'#'; E2 = 1\n", []),
    ("variables named like labels", "E1, E2 = f(x)\nR1 = E2 + 1\n", []),
    ("label in a string", 's = "text # R1"\n', []),
    ("label inside a string with a hash", 's = "a # b R1"\n', []),
    ("label in a docstring with a hash",
     '"""\nA docstring with R1 and a # sign.\n"""\nf() = 1\n', []),
    ("code after a character literal #", "c = '#'; E2 = 1\n", []),
    ("adjoint, comment without a label, label as code afterwards",
     "x = A' * B  # plain comment\nx = A'; M1 = 2\n", []),
    ("backtick command with a #", "run(`echo # R1`)\n", []),
    ("triple backtick command with a #", "run(```echo # R1```)\n", []),
    ("CPU name", "# measured on Apple M3 and Apple M2\n", []),
    ("label in a string after an interpolation",
     's = "$(x) # R1"\n', []),
    ("label in a string inside an interpolation", 's = "$(f("# R1"))"\n', []),
    ("label inside a raw string macro", 'r = raw"\\" # R1"\n', []),
    ("no comment at all", "function f(x)\n  x + 1\nend\n", []),
    ("label-like words that are not labels",
     "# R10x, XR1, Apple, T3rd, ER1 and E1x\n", []),
    # the exemption does not hide a label next to a CPU name
    ("label next to a CPU name", "# Apple M3 and R1\n", [(1, "R1")]),
    ("M3 not after Apple", "# a machine M3\n", [(1, "M3")]),
]


def self_test():
    failures = 0
    for name, source, expected in SELF_TEST_CASES:
        got = find_labels(source)
        if got != expected:
            failures += 1
            print(f"FAIL {name}: expected {expected}, got {got}")
        else:
            print(f"ok   {name}")
    # the whole tree: message format, exit status, relative paths, nested directories
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "src" / "functions").mkdir(parents=True)
        (tmp / "src" / "a.jl").write_text("x = 1\n# fine\n", encoding="utf-8")
        (tmp / "src" / "functions" / "b.jl").write_text("y = 2\n  # R1 here\n", encoding="utf-8")
        (tmp / "test").mkdir()
        (tmp / "test" / "c.jl").write_text("# R1 in a test comment is outside src/\n", encoding="utf-8")
        nfiles, messages = check_tree(tmp)
        want = [MESSAGE.format(path="src/functions/b.jl", line=2, token="R1")]
        if nfiles != 2 or messages != want:
            failures += 1
            print(f"FAIL tree with a label: expected 2 files and {want}, got {nfiles} and {messages}")
        else:
            print("ok   tree with a label (exit 1)")
        (tmp / "src" / "functions" / "b.jl").write_text("y = 2\n  # fine\n", encoding="utf-8")
        nfiles, messages = check_tree(tmp)
        if nfiles != 2 or messages:
            failures += 1
            print(f"FAIL tree without a label: got {nfiles} and {messages}")
        else:
            print("ok   tree without a label (exit 0)")
    print(f"check_source_labels self-test: {len(SELF_TEST_CASES) + 2 - failures} passed, {failures} failed")
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(ROOT),
                        help="repository root whose src/ is scanned (default: the parent of docs/)")
    parser.add_argument("--self-test", action="store_true",
                        help="run the built-in cases and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    return run(args.root)


if __name__ == "__main__":
    sys.exit(main())
