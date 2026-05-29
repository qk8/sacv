"""ESLint + tsc type-check runner for TypeScript projects."""
from __future__ import annotations
import subprocess, json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class LintResult:
    passed: bool
    violations: list[dict]
    raw_output: str

def run_eslint(repo_root: Path) -> LintResult:
    result = subprocess.run(
        ["npx", "eslint", ".", "--format", "json", "--max-warnings", "0"],
        cwd=str(repo_root), capture_output=True, text=True, timeout=120,
    )
    try:
        raw = json.loads(result.stdout)
        violations = [
            {"file": f["filePath"], "line": m["line"], "message": m["message"],
             "rule": m.get("ruleId", "?")}
            for f in raw for m in f.get("messages", [])
        ]
    except Exception:
        violations = []
    return LintResult(passed=result.returncode == 0, violations=violations,
                      raw_output=result.stdout[:3000])

def run_tsc(repo_root: Path) -> LintResult:
    result = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=str(repo_root), capture_output=True, text=True, timeout=120,
    )
    violations = [{"message": l} for l in result.stdout.splitlines() if "error TS" in l]
    return LintResult(passed=result.returncode == 0, violations=violations,
                      raw_output=result.stdout[:3000])
