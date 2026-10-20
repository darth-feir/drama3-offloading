"""Load config.toml, merge CLI overrides, hand back a plain object.

Three places a setting can come from, in increasing priority:

    the defaults below  ->  config.toml  ->  a command-line flag

which means a config file never has to be complete, and a flag always wins
for a one-off run without editing anything.

WHY A FILE AT ALL
-----------------
The quality checks are the settings that actually need tuning, and they
need tuning *per document*. Measured on a technical document, the content
check fired 9 times in 201 pieces where the genuine failure rate looks like
zero -- STT mangles BACPAC into "backpack" and a piece ending on one takes
a tail-weighted hit. On a novel the same check caught two real failures
that nothing else could see. Same code, opposite verdicts, and the only
difference is the text. That is a config file, not a constant.

TOML because it takes comments. Every threshold here was either measured or
guessed, and the file says which -- that is worth more than the number.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

HOME = Path(__file__).resolve().parent.parent

# Mirrors config.toml. Kept here as well so the server runs with no config
# file present, and so a partial file does not have to repeat the rest.
DEFAULTS: dict = {
    "voice": {"ref": None, "ref_text": None, "seconds": 56.0,
              "snap_to_silence": True},
    "sampling": {"temperature": 1.0, "top_p": 0.9, "top_k": 4097},
    "splitting": {"para_max_bytes": 400, "max_new_tokens": 600},
    "checks": {
        "max_attempts": 3,
        "duration": {"enabled": True, "min_ratio": 0.55, "max_ratio": 2.2,
                     "min_seconds": 1.5},
        "content": {"enabled": True, "min_similarity": 0.75, "min_words": 8,
                    "tail_fraction": 0.25},
    },
    "post": {
        "trim_trailing_silence": {"enabled": True, "pad_seconds": 0.15},
        "clamp_internal_silence": {"enabled": True, "max_gap": 0.5,
                                   "long_gap": 1.2},
    },
    "server": {"host": "127.0.0.1", "port": 8123},
}

# Only these may appear in a file. A typo in a config is otherwise silent --
# you set `min_similarty`, nothing complains, and the check runs at the
# default while you conclude it does not work.
KNOWN = {
    ("voice", k) for k in DEFAULTS["voice"]
} | {
    ("sampling", k) for k in DEFAULTS["sampling"]
} | {
    ("splitting", k) for k in DEFAULTS["splitting"]
} | {
    ("checks", "max_attempts"),
} | {
    ("checks.duration", k) for k in DEFAULTS["checks"]["duration"]
} | {
    ("checks.content", k) for k in DEFAULTS["checks"]["content"]
} | {
    ("post.trim_trailing_silence", k)
    for k in DEFAULTS["post"]["trim_trailing_silence"]
} | {
    ("post.clamp_internal_silence", k)
    for k in DEFAULTS["post"]["clamp_internal_silence"]
} | {
    ("server", k) for k in DEFAULTS["server"]
}


def _deep_merge(base: dict, over: dict, path: str = "") -> list[str]:
    """Merge `over` into `base` in place. Returns unknown keys found."""
    unknown = []
    for key, value in over.items():
        here = f"{path}.{key}" if path else key
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            unknown += _deep_merge(base[key], value, here)
        elif (path, key) in KNOWN or (here, key) in KNOWN:
            base[key] = value
        elif isinstance(value, dict):
            unknown.append(here)
        else:
            unknown.append(here)
    return unknown


def load(path: Path | str | None = None, strict: bool = True) -> SimpleNamespace:
    """Defaults, overlaid with a TOML file if there is one.

    `path` None means ./config.toml if it exists, otherwise pure defaults --
    so a fresh checkout runs without anyone writing a config first.
    """
    import copy
    merged = copy.deepcopy(DEFAULTS)
    chosen = Path(path) if path else (HOME / "config.toml")
    if path and not chosen.is_file():
        raise SystemExit(f"no config file at {chosen}")
    if chosen.is_file():
        with open(chosen, "rb") as handle:
            unknown = _deep_merge(merged, tomllib.load(handle))
        if unknown:
            message = (f"{chosen}: unknown setting(s): {', '.join(sorted(unknown))}")
            if strict:
                raise SystemExit(
                    message + "\nA typo in a config is otherwise silent: the "
                    "setting is ignored, the default applies, and you conclude "
                    "the feature does not work.")
            print(f"warning: {message}")
    merged["_path"] = str(chosen) if chosen.is_file() else "(defaults only)"
    return _namespace(merged)


def _namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**{
        k: _namespace(v) if isinstance(v, dict) else v for k, v in d.items()
    })


def describe(cfg: SimpleNamespace) -> str:
    """One line per check, for the startup log. Which are actually on."""
    bits = []
    bits.append(f"duration {'on' if cfg.checks.duration.enabled else 'OFF'}"
                f"({cfg.checks.duration.min_ratio}-{cfg.checks.duration.max_ratio})")
    bits.append(f"content {'on' if cfg.checks.content.enabled else 'OFF'}"
                f"(>={cfg.checks.content.min_similarity})")
    bits.append(f"trim {'on' if cfg.post.trim_trailing_silence.enabled else 'OFF'}")
    bits.append(f"clamp {'on' if cfg.post.clamp_internal_silence.enabled else 'OFF'}")
    return f"checks: {', '.join(bits)}, max_attempts {cfg.checks.max_attempts}"
