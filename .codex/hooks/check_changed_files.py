"""Run fast, read-only integrity checks after Codex edits files."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tokenize
from pathlib import Path


MAX_REPORTED_ISSUES = 40
SHARED_NOTE = Path("docs/experiment_notes/AMN_MINN_shared_reservoir_notes.md")
SHARED_EXACT_PATHS = {
    Path("generate_ecoli_iML1515_AMN_MINN_data.py"),
    Path("ecoli_iML1515_AMN_MINN_model_testing_trial.ipynb"),
    Path("ecoli_iML1515_MINN_AMN_model_testing_trial.ipynb"),
}


def run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def find_repo_root(cwd: Path) -> Path | None:
    result = run_git(cwd, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def changed_paths(root: Path) -> list[Path]:
    relative_paths: set[str] = set()
    commands = (
        ("diff", "--name-only", "--diff-filter=ACMR", "-z"),
        ("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"),
        ("ls-files", "--others", "--exclude-standard", "-z"),
    )
    for command in commands:
        result = run_git(root, *command)
        if result.returncode == 0:
            relative_paths.update(path for path in result.stdout.split("\0") if path)

    return sorted(
        (root / relative_path for relative_path in relative_paths),
        key=lambda path: str(path).casefold(),
    )


def check_python(path: Path, root: Path) -> str | None:
    try:
        with tokenize.open(path) as source_file:
            source = source_file.read()
        ast.parse(source, filename=str(path.relative_to(root)))
    except (OSError, SyntaxError, UnicodeError) as error:
        return f"{path.relative_to(root)}: {error}"
    return None


def check_notebook(path: Path, root: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as notebook_file:
            notebook = json.load(notebook_file)
        if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
            raise ValueError("expected a notebook object with a cells list")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return f"{path.relative_to(root)}: {error}"
    return None


def diff_issues(root: Path) -> list[str]:
    issues: list[str] = []
    for args in (("diff", "--check"), ("diff", "--cached", "--check")):
        result = run_git(root, *args)
        if result.returncode != 0 and result.stdout.strip():
            issues.extend(result.stdout.strip().splitlines())
    return issues


def is_shared_experiment_path(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if relative == SHARED_NOTE or relative.parts[0] in {".codex", "docs"}:
        return False
    if relative in SHARED_EXACT_PATHS:
        return True

    normalized_name = relative.name.casefold()
    has_shared_name = "amn_minn" in normalized_name or "minn_amn" in normalized_name
    if not has_shared_name:
        return False
    return relative.suffix.casefold() in {
        ".py", ".ipynb", ".json", ".toml", ".yaml", ".yml", ".md", ".txt"
    }


def documentation_coupling_issues(root: Path, paths: list[Path]) -> list[str]:
    relative_paths = {path.relative_to(root) for path in paths}
    related = [
        path.relative_to(root)
        for path in paths
        if is_shared_experiment_path(path, root)
    ]
    if related and SHARED_NOTE not in relative_paths:
        names = ", ".join(
            str(path) for path in sorted(related, key=lambda value: str(value).casefold())
        )
        return [
            f"Shared AMN/MINN experiment files changed without {SHARED_NOTE}: {names}"
        ]
    return []


def report(issues: list[str]) -> None:
    shown = issues[:MAX_REPORTED_ISSUES]
    if len(issues) > MAX_REPORTED_ISSUES:
        shown.append(f"...and {len(issues) - MAX_REPORTED_ISSUES} more issue(s)")
    details = "Post-edit integrity checks found:\n- " + "\n- ".join(shown)
    print(
        json.dumps(
            {
                "systemMessage": "Post-edit integrity checks found issues.",
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": details,
                },
            }
        )
    )


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        event = {}

    cwd = Path(event.get("cwd") or Path.cwd()).resolve()
    root = find_repo_root(cwd)
    if root is None:
        return 0

    paths = changed_paths(root)
    issues = diff_issues(root)
    issues.extend(documentation_coupling_issues(root, paths))
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix.lower() == ".py":
            issue = check_python(path, root)
        elif path.suffix.lower() == ".ipynb":
            issue = check_notebook(path, root)
        else:
            continue
        if issue:
            issues.append(issue)

    if issues:
        report(issues)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
