"""Shared raw-data scanning: match data/set*/*.jpg filenames to a class map
and drop exact-duplicate files (the raw set has several: set02 re-includes a
number of set01's photos verbatim). Used by prepare_dataset.py and
grain_pattern.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pipeline.config import KNOWN_VARIANT_SUFFIXES, RAW_DATA_DIRS, SPECIES_PREFIX

REPO_ROOT = Path(__file__).resolve().parent.parent


def stem_to_class(stem: str, class_prefixes: dict[str, list[str]]) -> str | None:
    s = stem
    if s.startswith(SPECIES_PREFIX):
        s = s[len(SPECIES_PREFIX):]
    for suf in KNOWN_VARIANT_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    for cls, prefixes in class_prefixes.items():
        if s in prefixes:
            return cls
    return None


def discover(class_prefixes: dict[str, list[str]]) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = {cls: [] for cls in class_prefixes}
    for raw_dir in RAW_DATA_DIRS:
        d = REPO_ROOT / raw_dir
        for f in sorted(d.glob("*.jpg")):
            cls = stem_to_class(f.stem, class_prefixes)
            if cls is not None:
                by_class[cls].append(f)
    return by_class


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def dedupe(paths: list[Path], verbose: bool = True) -> list[Path]:
    seen: dict[str, Path] = {}
    for p in paths:
        h = file_hash(p)
        if h not in seen:
            seen[h] = p
        elif verbose:
            print(f"  skipping duplicate of {seen[h].name}: {p}")
    return list(seen.values())
