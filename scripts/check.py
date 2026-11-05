"""Run all quality checks one after another and report a summary.

Every check runs even if an earlier one failed. Exit code is 1 if any check failed.
Usage: python scripts/check.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHECKS: list[tuple[str, list[str]]] = [
    ("ruff check", ["-m", "ruff", "check", "."]),
    ("ruff format --check", ["-m", "ruff", "format", "--check", "."]),
    ("mypy", ["-m", "mypy"]),
    ("pytest", ["-m", "pytest"]),
]


def run_check(args: list[str]) -> int:
    """Run `python <args>` in the repository root and return its exit code."""
    # Arguments come from the constant CHECKS list above, not from user input.
    return subprocess.run([sys.executable, *args], cwd=ROOT, check=False).returncode  # noqa: S603


def main() -> int:
    """Run every check and return 1 if at least one failed, otherwise 0."""
    results: list[tuple[str, int]] = []
    for name, args in CHECKS:
        sys.stdout.write(f"\n=== {name} ===\n")
        sys.stdout.flush()
        results.append((name, run_check(args)))

    sys.stdout.write("\n=== Summary ===\n")
    for name, code in results:
        status = "OK" if code == 0 else f"FAILED (exit code {code})"
        sys.stdout.write(f"{name}: {status}\n")
    return 0 if all(code == 0 for _, code in results) else 1


if __name__ == "__main__":
    sys.exit(main())
