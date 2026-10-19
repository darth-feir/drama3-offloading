"""Verify every `see notes/botnotes/<slug>.md` pointer resolves, and that no
botnote is orphaned.

Moving long explanations out of the code buys readability and takes on one
new failure mode: the pointer goes stale and the reader is sent nowhere.
Inline comments rot too, but they rot silently in a way nobody can check.
This is the check.

Fails on:
  * a reference to a botnote that does not exist
  * a botnote that nothing references  (delete it, or point at it)
  * a botnote missing its "What breaks if you change it back" section,
    which is the one part of the template that justifies the directory

  bin/run.sh bin/check_botnotes.py        (no GPU, no model, no venv deps)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent
BOTNOTES = HOME / "notes" / "botnotes"
REFERENCE = re.compile(r"notes/botnotes/([A-Za-z0-9._-]+\.md)")
SEARCH = ("bin/*.py", "bin/*.sh", "csrc/*.cu", "notes/*.md", "*.md")


def main() -> int:
    referenced: dict[str, list[str]] = {}
    for pattern in SEARCH:
        for path in sorted(HOME.glob(pattern)):
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for slug in REFERENCE.findall(text):
                referenced.setdefault(slug, []).append(
                    str(path.relative_to(HOME)))

    existing = {p.name for p in BOTNOTES.glob("*.md")} - {"README.md"}
    problems = []

    for slug, where in sorted(referenced.items()):
        if slug == "README.md":
            continue
        if slug not in existing:
            problems.append(
                f"BROKEN   notes/botnotes/{slug} referenced by "
                f"{', '.join(sorted(set(where)))} but does not exist")

    for slug in sorted(existing - set(referenced)):
        problems.append(
            f"ORPHAN   notes/botnotes/{slug} exists but nothing points at it")

    for slug in sorted(existing & set(referenced)):
        body = (BOTNOTES / slug).read_text()
        if "What breaks if you change it back" not in body:
            problems.append(
                f"NO-WHY   notes/botnotes/{slug} has no "
                f"'What breaks if you change it back' section")

    live = sorted(set(referenced) & existing)
    print(f"{len(live)} botnote(s) referenced from "
          f"{len({w for ws in referenced.values() for w in ws})} file(s)")
    for slug in live:
        print(f"  {slug:<40s} <- {', '.join(sorted(set(referenced[slug])))}")

    if problems:
        print()
        for problem in problems:
            print(f"  {problem}")
        print(f"\nFAILED: {len(problems)} problem(s)")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
