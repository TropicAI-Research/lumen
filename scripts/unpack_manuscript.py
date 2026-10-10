#!/usr/bin/env python3
"""Restore the verified paper sources locally, without overwriting edits."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "manuscript"


def unpack(destination=PAPER, root=ROOT):
    destination = Path(destination)
    assert not destination.is_symlink(), "A manuscript destination cannot be a symlink."
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Refusing to overwrite existing manuscript sources: " + str(destination))
    root = Path(root)
    manifest = json.loads((root / "release/build_manifest.json").read_text())
    archive = root / "release/lumen-bench-overleaf.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["overleaf_archive_sha256"]
    rows = manifest["source_files"]
    with zipfile.ZipFile(archive) as bundle:
        assert len(bundle.namelist()) == len(rows)
        assert set(bundle.namelist()) == {r["archive_path"] for r in rows}
        verified = []
        for row in rows:
            name = PurePosixPath(row["archive_path"])
            assert not name.is_absolute() and ".." not in name.parts
            assert "\\" not in str(name)
            content = bundle.read(str(name))
            assert hashlib.sha256(content).hexdigest() == row["sha256"], str(name)
            relative = Path("neurips_2026.tex") if str(name) == "main.tex" else Path(name)
            verified.append((relative, content))
    # All paths and bytes are validated before any file is created.
    destination.mkdir(parents=True, exist_ok=True)
    for relative, content in verified:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(content)
    return destination


def ensure_manuscript():
    """Builders preserve an existing working manuscript or unpack the release."""
    if (PAPER / "neurips_2026.tex").is_file():
        assert not PAPER.is_symlink()
        return PAPER
    return unpack()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PAPER)
    args = parser.parse_args()
    print(unpack(args.directory))


if __name__ == "__main__":
    main()
