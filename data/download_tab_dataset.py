"""Helper script to clone the Text Anonymization Benchmark dataset.

This script wraps a `git clone` call and optionally performs a `git pull`.
Use sparse checkout to pull only the files you care about (e.g., select ECHR splits).
Run Git LFS commands manually if you need large assets.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


TAB_REPO_URL = "https://github.com/NorskRegnesentral/text-anonymization-benchmark.git"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Invoke a shell command and exit early on failure."""
    print("$", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)


def run_optional(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run a command but allow it to fail (used for sparse-checkout init)."""
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=False)


def ensure_git_available() -> None:
    if shutil.which("git"):
        return
    print("Error: `git` not found in PATH. Install Git and re-run.", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    default_dest = Path(__file__).resolve().parent.parent / "external" / "text-anonymization-benchmark"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=default_dest,
        help=f"Target directory for the TAB repository (default: {default_dest})",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional branch or tag to clone (default: repository default)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="If the repository already exists, run `git pull` instead of cloning.",
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        default=[],
        help="Relative file path to include via sparse checkout. Repeat for multiple files.",
    )
    return parser.parse_args()


def configure_sparse_checkout(dest: Path, files: list[str]) -> None:
    if not files:
        return
    run_optional(["git", "sparse-checkout", "init", "--no-cone"], cwd=dest)
    run(["git", "sparse-checkout", "set", *files], cwd=dest)


def clone_or_update(dest: Path, branch: str | None, update: bool, files: list[str]) -> None:
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if not (dest / ".git").exists():
            print(f"Error: destination {dest} exists but is not a git repository.", file=sys.stderr)
            sys.exit(1)
        if update:
            run(["git", "pull"], cwd=dest)
        else:
            print(f"Repository already present at {dest}. Use --update to pull latest changes.")
        if files:
            configure_sparse_checkout(dest, files)
    else:
        cmd = ["git", "clone"]
        if files:
            cmd.extend(["--filter=blob:none", "--sparse"])
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([TAB_REPO_URL, str(dest)])
        run(cmd)
        if files:
            configure_sparse_checkout(dest, files)


def main() -> None:
    ensure_git_available()
    args = parse_args()
    clone_or_update(args.dest.resolve(), args.branch, args.update, args.files)


if __name__ == "__main__":
    main()
