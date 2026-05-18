"""Download or stage a development subset of the MFCAD++ dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import requests


MFCAD_REPO_URL = "https://github.com/hiearchy/MFCAD-plus-plus"
TARGET_DIR = Path("data") / "raw" / "mfcad_plus_plus"
MANIFEST = TARGET_DIR / "manifest.json"


def _step_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".stp", ".step"}
    )


def _valid_step_header(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(64).lstrip().startswith(b"ISO-10303-21")
    except OSError:
        return False


def _write_manifest(files: list[Path], source: str) -> None:
    classes: dict[str, int] = {}
    for path in files:
        class_name = path.parent.name if path.parent != TARGET_DIR else "unclassified"
        classes[class_name] = classes.get(class_name, 0) + 1

    manifest = {
        "total_files": len(files),
        "download_date": date.today().isoformat(),
        "source": source,
        "classes": dict(sorted(classes.items())),
    }
    with MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _github_releases_available() -> bool:
    url = "https://api.github.com/repos/hiearchy/MFCAD-plus-plus/releases/latest"
    try:
        response = requests.get(url, timeout=20)
    except requests.RequestException:
        return False
    return response.status_code == 200 and bool(response.json().get("assets"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a representative MFCAD++ STEP subset."
    )
    parser.add_argument("--subset", type=int, default=500)
    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    existing = _step_files(TARGET_DIR)
    if existing:
        valid = [path for path in existing if _valid_step_header(path)]
        _write_manifest(valid, "MFCAD++")
        print(f"Found {len(existing)} STEP files in {TARGET_DIR}; skipping download.")
        print(f"Validated {len(valid)} files and wrote {MANIFEST}.")
        return 0

    if _github_releases_available():
        print("MFCAD++ release assets were found, but automatic asset layout is unknown.")

    print("Automatic MFCAD++ download is unavailable from the public repo metadata.")
    print("Manual setup:")
    print(f"  1. Download the MFCAD++ dataset from {MFCAD_REPO_URL}")
    print(f"  2. Copy or extract STEP files into {TARGET_DIR}")
    print(f"  3. Re-run: python scripts/download_mfcad.py --subset {args.subset}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
