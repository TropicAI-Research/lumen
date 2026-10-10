#!/usr/bin/env python3
"""Compile/package only the current preprint and its explicit TeX inputs.

No submission, network access, system installation, or unrelated workspace
files. The source archive excludes build logs, auxiliary files, and the PDF.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.unpack_manuscript import PAPER, ensure_manuscript


def collect_sources(paper=None):
    paper = (paper or PAPER).resolve()
    pending = [Path("neurips_2026.tex")]
    selected = {Path("references.bib"), Path("neurips_2026.sty")}
    while pending:
        relative = pending.pop()
        if relative in selected:
            continue
        source = paper / relative
        assert source.resolve().is_relative_to(paper) and not source.is_symlink()
        content = source.read_text()
        assert "Working copy" not in content
        selected.add(relative)
        # Ignore full-line comments, without corrupting escaped percent signs.
        content = "\n".join(line for line in content.splitlines() if not line.lstrip().startswith("%"))
        for name in re.findall(r"\\(?:input|InputIfFileExists)\{([^}]+)\}", content):
            target = Path(name)
            pending.append(target if target.suffix else target.with_suffix(".tex"))
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content):
            target = Path(name)
            choices = [target] if target.suffix else [target.with_suffix(ext) for ext in [".pdf", ".png", ".jpg"]]
            asset = next((p for p in choices if (paper / p).is_file()), None)
            assert asset is not None, "Missing figure: " + name
            assert (paper / asset).resolve().is_relative_to(paper) and not (paper / asset).is_symlink()
            selected.add(asset)
    for required in ["generated/evaluation_coverage.tex", "generated/benign_tables.tex", "generated/benign_detail.tex"]:
        assert Path(required) in selected and (paper / required).is_file(), required
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=PAPER)
    parser.add_argument("--output-stem", default="lumen-bench")
    args = parser.parse_args()
    if args.paper == PAPER:
        ensure_manuscript()
    paper = args.paper.resolve()
    assert paper.is_relative_to(ROOT) and paper.is_dir(), "Use a paper directory inside this project."
    assert re.fullmatch(r"[A-Za-z0-9_-]+", args.output_stem), "Invalid output stem"
    sources = collect_sources(paper)
    build = Path(tempfile.mkdtemp(prefix="lumen-arxiv-build-"))
    entries = []
    for relative in sources:
        target = Path("main.tex") if str(relative) == "neurips_2026.tex" else relative
        (build / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paper / relative, build / target)
        entries.append({"source": str((paper / relative).relative_to(ROOT)), "archive_path": str(target),
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
    # Additional floats can move appendix labels after the usual three passes.
    # Stop only when LaTeX reports stable cross-references, with a finite cap.
    for _ in range(3):
        if "Rerun to get cross-references right" not in (build / "main.log").read_text():
            break
        command = engine + ["-interaction=batchmode", "-halt-on-error", "main.tex"]
        run = subprocess.run(command, cwd=build, capture_output=True, text=True)
        log.append({"command": command, "returncode": run.returncode})
        if run.returncode:
            raise SystemExit(f"Build failed; inspect {build}/main.log.")
    text = (build / "main.log").read_text()
    fatal = [line for line in text.splitlines() if "Missing character:" in line or "Overfull" in line or
             "undefined references" in line or "undefined citations" in line or
             "Rerun to get cross-references right" in line or line.startswith("!")]
    assert not fatal, "Unresolved PDF diagnostics: " + repr(fatal)
    extract = subprocess.run(["pdftotext", "-layout", str(build / "main.pdf"), "-"], capture_output=True, text=True, check=True).stdout
    assert "Working copy" not in extract and "??" not in extract
    statistics = json.loads((ROOT / "analysis/revision/corrected_statistics.json").read_text())["audit"]
    sdg_percent = 100 * statistics["counts"]["sdg"] / statistics["counts"]["l2"]
    assert f"{sdg_percent:.2f}%" in extract and f"{statistics['n_cells']:,}" in extract
    for marker in ["anonymous.4open.science", "datasets/lumen-bench/lumen-bench", "Submitted to", "not ready for publication"]:
        assert marker not in extract, marker
    output = ROOT / "release"
    output.mkdir(exist_ok=True)
    pdf = output / (args.output_stem + ".pdf")
    archive = output / (args.output_stem + "-arxiv.tar.gz")
    overleaf = output / (args.output_stem + "-overleaf.zip")
    shutil.copy2(build / "main.pdf", pdf)
    entries.append({"source": "generated_by_bibtex", "archive_path": "main.bbl",
                    "sha256": hashlib.sha256((build / "main.bbl").read_bytes()).hexdigest()})
    with tarfile.open(archive, "w:gz") as bundle:
        for row in entries:
            bundle.add(build / row["archive_path"], arcname=row["archive_path"], recursive=False)
    with zipfile.ZipFile(overleaf, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for row in entries:
            bundle.write(build / row["archive_path"], arcname=row["archive_path"])
    with zipfile.ZipFile(overleaf) as bundle:
        assert bundle.testzip() is None
        assert sorted(bundle.namelist()) == sorted(row["archive_path"] for row in entries)
        for row in entries:
            assert hashlib.sha256(bundle.read(row["archive_path"])).hexdigest() == row["sha256"]
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    manifest = {"engine": "XeLaTeX", "local_engine_version": subprocess.run([engine[0], "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "arxiv_target": "XeLaTeX, TeX Live 2025", "tested_on_arxiv_server": False,
        "commands": log, "source_files": entries, "pages": pages,
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "overleaf_archive_sha256": hashlib.sha256(overleaf.read_bytes()).hexdigest(),
        "missing_glyphs": 0, "overfull_boxes": 0, "unresolved_references": 0,
        "visual_review_required": True, "arxiv_submitted": False}
    manifest_name = "build_manifest.json" if args.output_stem == "lumen-bench" else args.output_stem + "-build-manifest.json"
    (output / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pdf": str(pdf.relative_to(ROOT)), "sources": str(archive.relative_to(ROOT)), "overleaf": str(overleaf.relative_to(ROOT)),
        "pages": pages, "source_files": len(entries), "build_directory": str(build), "submitted": False}))


if __name__ == "__main__":
    main()
