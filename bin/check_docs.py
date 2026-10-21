#!/usr/bin/env python
"""Check the documentation still describes the code. No GPU, no model, no deps.

    venv/bin/python bin/check_docs.py

WHY THIS EXISTS
---------------
Documentation in this repo has drifted three times, and each time a human
reading found it rather than anything automatic:

  * API.md said `speed` was "accepted and ignored" for days after it became
    a measured ffmpeg post-process -- in the one file aimed at people who
    should never have to read our source.
  * bin/paths.py printed `huggingface-cli download`, renamed two releases
    earlier to `hf`.
  * README's own smoke command passes `--text-file document.txt`, and no
    such file is in the repo. TWO clean-room builds missed that, because
    both supplied their own text.

What this CAN check is whether the docs name things that exist: flags,
config keys, routes, files. It is a spell-checker against the code, not a
fact-checker -- it would have caught the third of those three and neither
of the first two, because a wrong *claim* about a flag that exists is not
a drift this can see. Said plainly so nobody trusts a PASS too far.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent
DOCS = ["README.md", "API.md", "CONTRIBUTING.md"]
problems: list[str] = []
checked = 0


def read(name: str) -> str:
    p = HOME / name
    return p.read_text() if p.is_file() else ""


def code_text() -> str:
    """Every python source plus the control script, concatenated."""
    out = []
    # ⚠️ Shell scripts too, not just *.py. This scanned python plus
    # drama3ctl and therefore could not see flags defined in setup.sh --
    # it reported --yes and --no-weights as undocumented inventions when
    # both were real. A checker that only half-knows the codebase
    # produces false alarms, which is how checkers get ignored.
    for p in sorted((HOME / "bin").glob("*.py")):
        out.append(p.read_text())
    for p in sorted((HOME / "bin").glob("*.sh")):
        out.append(p.read_text())
    ctl = HOME / "bin" / "drama3ctl"
    if ctl.is_file():
        out.append(ctl.read_text())
    return "\n".join(out)


def check_flags(docs: dict[str, str], code: str) -> None:
    """Every --flag named in the docs should appear in the code."""
    global checked
    # Flags the docs legitimately mention that belong to other programs.
    foreign = {
        "--no-deps", "--upgrade", "--version", "--help", "--limit-rate",
        "--stt.models_dir", "--stt.sherpa.model_id", "--run", "-p",
        "--out-dir-is-not-a-flag",
    }
    for name, text in docs.items():
        for flag in sorted(set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]{2,}", text))):
            if flag in foreign:
                continue
            checked += 1
            dest = flag[2:].replace("-", "_")
            if flag in code or f'"{dest}"' in code or f"'{dest}'" in code:
                continue
            problems.append(f"{name}: flag {flag} is not in any source file")


def check_config_keys(docs: dict[str, str]) -> None:
    """Every `[section] key` the docs mention should be a real config key."""
    global checked
    sys.path.insert(0, str(HOME / "bin"))
    try:
        import config
    except Exception as exc:                        # noqa: BLE001
        problems.append(f"cannot import bin/config.py: {exc}")
        return
    known = {f"{a}.{b}" for a, b in config.KNOWN}
    known |= {b for _, b in config.KNOWN}
    for name, text in docs.items():
        for sect, key in re.findall(r"`?\[(\w+)\]`?\s+`?(\w+)`?", text):
            checked += 1
            if f"{sect}.{key}" in known or key in known:
                continue
            problems.append(f"{name}: config key [{sect}] {key} is unknown")


def check_routes(docs: dict[str, str], code: str) -> None:
    """Every HTTP route documented should exist in the server."""
    global checked
    for name, text in docs.items():
        for route in sorted(set(re.findall(r"(/v1/[a-z/_]+|/health|/tts\w*)", text))):
            checked += 1
            if route in code:
                continue
            problems.append(f"{name}: route {route} is not served")


def check_referenced_files(docs: dict[str, str]) -> None:
    """Files the docs tell you to run or read should be in the repo.

    This is the check that catches `--text-file document.txt` when no
    document.txt was ever committed.
    """
    global checked
    exts = "toml|txt|md|py|sh"
    # ⚠️ Quoted names FIRST, because filenames here contain spaces --
    # "text for testing quote from Dune.txt". The bare-word pattern below
    # matched `Dune.txt` out of the middle of it and reported a missing
    # file that was sitting in the repo.
    quoted = re.compile(rf'["`]([^"`\n]+\.(?:{exts}))["`]')
    bare = re.compile(rf"(?:`|\s)((?:bin|notes|csrc)/[\w.-]+|"
                      rf"[\w-]+\.(?:{exts}))(?:`|\s|$)")
    for name, text in docs.items():
        # Files the docs tell you to CREATE are not files the repo should
        # ship. `... > list.txt` was reported missing when the command
        # two lines above writes it.
        created = set(re.findall(r"(?:>|-o|--output)\s+([\w./-]+)", text))
        names = set(quoted.findall(text))
        # a bare match that is the tail of a quoted name is that name
        names |= {r for r in bare.findall(text)
                  if not any(q.endswith(r) and q != r for q in names)}
        for ref in sorted(names):
            # A backticked COMMAND whose last word is a path -- `source
            # bin/env.sh` -- is not a reference to a file called
            # "source bin/env.sh". Take the path off the end.
            if " " in ref and not (HOME / ref).exists():
                ref = ref.rsplit(" ", 1)[-1]
            # Upstream files, named to say where a bug lives, not shipped here.
            if ref.startswith(("http", "fish_speech", "fish_omni")):
                continue
            if ref in created or Path(ref).name in created:
                continue
            # DOC.txt, YOURFILE.wav: an ALL-CAPS stem is the convention in
            # these docs for "substitute your own", not a repo file.
            if Path(ref).stem.isupper():
                continue
            checked += 1
            if (HOME / ref).exists():
                continue
            problems.append(f"{name}: references {ref}, which is not in the repo")


def main() -> int:
    docs = {n: read(n) for n in DOCS if read(n)}
    if not docs:
        print("no documentation found")
        return 1
    code = code_text()
    check_flags(docs, code)
    check_config_keys(docs)
    check_routes(docs, code)
    check_referenced_files(docs)

    for p in problems:
        print(f"  {p}")
    print()
    print(f"{checked} reference(s) checked across {len(docs)} document(s)")
    if problems:
        print(f"{len(problems)} problem(s) -- FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
