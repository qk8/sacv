# NOT YET WIRED — see ARCH-003.
"""Checkstyle + SpotBugs runner for Java projects. Not connected to preflight_node."""
from __future__ import annotations
import subprocess
from pathlib import Path
from dataclasses import dataclass

@dataclass
class LintResult:
    passed: bool
    violations: list[dict]
    raw_output: str

def run_checkstyle(repo_root: Path, config_path: str = "checkstyle.xml") -> LintResult:
    result = subprocess.run(
        ["mvn", "checkstyle:check", "-q", f"-Dcheckstyle.config.location={config_path}"],
        cwd=str(repo_root), capture_output=True, text=True, timeout=120,
    )
    violations = _parse_checkstyle(result.stdout + result.stderr)
    return LintResult(passed=result.returncode == 0, violations=violations,
                      raw_output=result.stdout[:3000])

def _parse_checkstyle(output: str) -> list[dict]:
    findings = []
    for line in output.splitlines():
        if "[WARN]" in line or "[ERROR]" in line:
            findings.append({"message": line.strip()})
    return findings[:50]
