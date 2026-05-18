"""Validate the local RPP MVP development environment."""

from __future__ import annotations

import importlib
import os


REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("trimesh", "trimesh"),
    ("torch", "torch"),
    ("scipy", "scipy"),
    ("tqdm", "tqdm"),
]

OCC_PACKAGES = [
    ("OCC.Core.STEPControl", "pythonocc-core"),
    ("cadquery", "cadquery"),
]

REQUIRED_DIRECTORIES = [
    os.path.join("data", "raw"),
    os.path.join("data", "processed"),
]


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def check_imports() -> list[tuple[str, bool]]:
    """Return package import status for required non-STEP dependencies."""
    return [(display, _can_import(module)) for module, display in REQUIRED_PACKAGES]


def check_step_backend() -> str | None:
    """Return the active STEP backend name, preferring OCC over cadquery."""
    for module, display in OCC_PACKAGES:
        if _can_import(module):
            return "occ" if display == "pythonocc-core" else "cadquery"
    return None


def check_directories() -> None:
    """Create required data directories if absent."""
    for path in REQUIRED_DIRECTORIES:
        os.makedirs(path, exist_ok=True)


def main() -> None:
    print("Checking dependencies...")
    import_results = check_imports()
    for name, ok in import_results:
        print(f"  {name:<14}{'PASS' if ok else 'FAIL'}")

    print()
    print("Checking STEP backend...")
    backend = check_step_backend()
    occ_ok = backend == "occ"
    cq_ok = backend == "cadquery"
    print(f"  {'pythonocc-core':<14}{'PASS' if occ_ok else 'FAIL'}")
    print(f"  {'cadquery':<14}{'PASS' if cq_ok else 'FAIL'}")
    if backend == "occ":
        print("  preferred backend active")
    elif backend == "cadquery":
        print("  fallback backend active")

    print()
    print("Checking directories...")
    check_directories()
    for path in REQUIRED_DIRECTORIES:
        print(f"  {path + '/':<18}PASS")

    print()
    if all(ok for _, ok in import_results) and backend is not None:
        print("Environment OK - ready for Phase 1.")
    else:
        print("Environment INCOMPLETE")


if __name__ == "__main__":
    main()
