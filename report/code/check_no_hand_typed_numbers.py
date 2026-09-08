#!/usr/bin/env python3
"""
Scan report/sections/*.tex for numeric literals not backed by a \\Metric... macro
or an explicit `% MANUAL:` allowlist comment (for genuine constants like
hyperparameters), and separately warn if a manifest's generated_at predates its
source notebook's mtime (catches "reran the notebook, forgot to regenerate").

Run manually (no CI):
    python report/code/check_no_hand_typed_numbers.py

Exits non-zero if it finds any un-allowlisted bare number, so it can gate a
"finalize this phase" checklist even without CI.
"""

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SECTIONS_DIR = REPO_ROOT / "report" / "sections"
MANIFEST_DIR = REPO_ROOT / "analysis" / "results_manifest"
NOTEBOOK_DIR = REPO_ROOT / "analysis" / "notebooks"
GENERATED_MACROS = SECTIONS_DIR / "_generated_macros.tex"

# A bare number: digits with an optional decimal point, not part of a LaTeX
# command/label/ref/cite, not a section/table/figure counter reference.
NUMBER_RE = re.compile(r"(?<![A-Za-z\\])\d+\.?\d*")
MANUAL_ALLOWLIST_MARKER = "% MANUAL:"


def line_has_manual_allowlist(line: str) -> bool:
    return MANUAL_ALLOWLIST_MARKER in line


# Matches a structural LaTeX command's full invocation, including any
# [optional] args and its {required} arg(s) -- e.g. \ref{sec:foo},
# \includegraphics[width=\linewidth]{figures/x.pdf}. Digits legitimately
# appear inside these (a label slug, a width fraction) and aren't "hand-typed
# results"; digits elsewhere on the SAME line (regular prose) still are.
STRUCTURAL_COMMAND_RE = re.compile(
    r"\\(?:label|ref|cite|includegraphics|input|cref)\b(?:\[[^\]]*\])?\{[^}]*\}"
)


def find_bare_numbers(tex_path: Path):
    findings = []
    for lineno, line in enumerate(tex_path.read_text().splitlines(), start=1):
        if line_has_manual_allowlist(line):
            continue
        # Strip structural command invocations (not the whole line -- a line
        # can legitimately mix a \ref{...} with unrelated hand-typed prose
        # numbers that still need checking) before scanning for bare numbers.
        scannable = STRUCTURAL_COMMAND_RE.sub("", line)
        for m in NUMBER_RE.finditer(scannable):
            findings.append((lineno, line.strip(), m.group()))
    return findings


def check_sections():
    all_findings = {}
    for tex_path in sorted(SECTIONS_DIR.glob("*.tex")):
        if tex_path.name == "_generated_macros.tex":
            continue
        findings = find_bare_numbers(tex_path)
        if findings:
            all_findings[tex_path] = findings
    return all_findings


def check_stale_manifests():
    warnings = []
    for mf in sorted(MANIFEST_DIR.glob("*.yaml")):
        with open(mf) as f:
            manifest = yaml.safe_load(f)
        phase = manifest.get("phase", mf.stem)
        generated_at = manifest.get("generated_at")
        # Find a notebook whose stem matches or starts with the phase slug.
        candidates = list(NOTEBOOK_DIR.glob(f"{phase}*.ipynb")) + \
                     list(NOTEBOOK_DIR.glob(f"{phase.split('_')[0]}_*.ipynb"))
        if not candidates:
            continue
        nb_path = candidates[0]
        nb_mtime = nb_path.stat().st_mtime
        try:
            from datetime import datetime
            gen_dt = datetime.fromisoformat(generated_at)
            gen_ts = gen_dt.timestamp()
        except (ValueError, TypeError):
            continue
        if gen_ts < nb_mtime:
            warnings.append((mf, nb_path))
    return warnings


def main():
    exit_code = 0

    findings = check_sections()
    if findings:
        exit_code = 1
        print("Bare numeric literals found (not backed by a \\Metric macro or % MANUAL: comment):")
        for tex_path, items in findings.items():
            print(f"\n  {tex_path.relative_to(REPO_ROOT)}:")
            for lineno, line, number in items:
                print(f"    L{lineno}: '{number}' in: {line[:100]}")
    else:
        print("No un-allowlisted bare numbers found in report/sections/*.tex.")

    stale = check_stale_manifests()
    if stale:
        print("\nWARNING: manifest older than its source notebook (reran notebook, forgot to "
              "regenerate manifest, or vice versa):")
        for mf, nb_path in stale:
            print(f"  {mf.relative_to(REPO_ROOT)}  (older than {nb_path.relative_to(REPO_ROOT)})")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
