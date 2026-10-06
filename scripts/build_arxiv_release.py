#!/usr/bin/env python3
"""Compile/package only the current preprint and its explicit TeX inputs.

No submission, network access, system installation, or unrelated workspace
files. The source archive excludes build logs, auxiliary files, and the PDF.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "Lumen_bench_For_NeurIPS_2026 (1)"


def collect_sources():
    pending = [Path("neurips_2026.tex")]
    selected = {Path("references.bib"), Path("neurips_2026.sty")}
    while pending:
        relative = pending.pop()
        if relative in selected:
            continue
        source = PAPER / relative
        assert source.resolve().is_relative_to(PAPER) and not source.is_symlink()
        content = source.read_text()
        assert "Working copy" not in content
        selected.add(relative)
        # Ignore full-line comments, without corrupting escaped percent signs.
        content = "\n".join(line for line in content.splitlines() if not line.lstrip().startswith("%"))
        for name in re.findall(r"\\(?:input|InputIfFileExists)\{([^}]+)\}", content):
            target = Path(name)
            pending.append(target if target.suffix else target.with_suffix(".tex"))
    for required in ["generated/repair_coverage.tex", "generated/benign_summary.tex", "generated/benign_detail.tex"]:
        assert Path(required) in selected and (PAPER / required).is_file(), required
    return sorted(selected)


def main():
    sources = collect_sources()
    build = Path(tempfile.mkdtemp(prefix="lumen-arxiv-build-"))
    entries = []
    for relative in sources:
        target = Path("main.tex") if str(relative) == "neurips_2026.tex" else relative
        (build / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAPER / relative, build / target)
        entries.append({"source": str((PAPER / relative).relative_to(ROOT)), "archive_path": str(target),
                        "sha256": hashlib.sha256((build / target).read_bytes()).hexdigest()})
    if shutil.which("xelatex"):
        engine = ["xelatex"]
    else:
        assert shutil.which("xetex"), "Install XeLaTeX before building."
        engine = ["xetex", "-fmt=xelatex", "-progname=xelatex"]
    log = []
    commands = [engine + ["-interaction=batchmode", "-halt-on-error", "main.tex"], ["bibtex", "main"],
                engine + ["-interaction=batchmode", "-halt-on-error", "main.tex"],
                engine + ["-interaction=batchmode", "-halt-on-error", "main.tex"]]
    for command in commands:
        run = subprocess.run(command, cwd=build, capture_output=True, text=True)
        log.append({"command": command, "returncode": run.returncode})
        if run.returncode:
            raise SystemExit(f"Build failed; inspect {build}/main.log. Command: {command[0]}")
    text = (build / "main.log").read_text()
    fatal = [line for line in text.splitlines() if "Missing character:" in line or "Overfull" in line or
             "undefined references" in line or "undefined citations" in line or line.startswith("!")]
    assert not fatal, "Unresolved PDF diagnostics: " + repr(fatal)
    extract = subprocess.run(["pdftotext", "-layout", str(build / "main.pdf"), "-"], capture_output=True, text=True, check=True).stdout
    assert "Working copy" not in extract and "??" not in extract
    assert "8.34%" in extract and "195,229" in extract
    for marker in ["anonymous.4open.science", "datasets/lumen-bench/lumen-bench", "Submitted to", "not ready for publication"]:
        assert marker not in extract, marker
    output = ROOT / "release"
    output.mkdir(exist_ok=True)
    pdf = output / "lumen-bench.pdf"
    archive = output / "lumen-bench-arxiv.tar.gz"
    shutil.copy2(build / "main.pdf", pdf)
    entries.append({"source": "generated_by_bibtex", "archive_path": "main.bbl",
                    "sha256": hashlib.sha256((build / "main.bbl").read_bytes()).hexdigest()})
    with tarfile.open(archive, "w:gz") as bundle:
        for row in entries:
            bundle.add(build / row["archive_path"], arcname=row["archive_path"], recursive=False)
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    manifest = {"engine": "XeLaTeX", "local_engine_version": subprocess.run([engine[0], "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "arxiv_target": "XeLaTeX, TeX Live 2025", "tested_on_arxiv_server": False,
        "commands": log, "source_files": entries, "pages": pages,
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "missing_glyphs": 0, "overfull_boxes": 0, "unresolved_references": 0,
        "visual_review_required": True, "arxiv_submitted": False}
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pdf": str(pdf.relative_to(ROOT)), "sources": str(archive.relative_to(ROOT)),
        "pages": pages, "source_files": len(entries), "build_directory": str(build), "submitted": False}))


if __name__ == "__main__":
    main()
