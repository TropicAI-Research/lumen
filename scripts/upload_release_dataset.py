#!/usr/bin/env python3
"""Upload an audited stage to the explicitly authorized personal HF dataset.

The destination must remain private. This never logs in globally, removes
remote files, or changes visibility. A parent commit guards against concurrent
edits. Credentials are loaded from the local ignored .env and never printed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_language_repairs import env_values, save
from scripts.stage_release import STATIC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    stage = args.directory.resolve()
    assert stage.parent == Path("/tmp") and stage.name.startswith("lumen-public-")
    security = json.loads((ROOT / "analysis/verification/release_candidate_security.json").read_text())
    assert not security["pattern_findings"]
    manifest = json.loads((ROOT / "analysis/verification/release_candidate_manifest.json").read_text())
    selected = [r for r in manifest if not args.static or r["path"].startswith(STATIC)]
    for row in selected:
        path = stage / row["path"]
        assert not path.is_symlink() and path.resolve().is_relative_to(stage)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"]
    print(json.dumps({"files_verified": len(selected), "destination": "Tropic-AI/lumen-bench",
                      "mode": "static" if args.static else "full", "execute": args.execute}), flush=True)
    if not args.execute:
        return
    os.environ["HF_HOME"] = "/tmp/lumen-hf-upload-cache"
    os.environ["HF_XET_CACHE"] = "/tmp/lumen-hf-upload-cache/xet"
    from huggingface_hub import HfApi
    token = env_values()["HF_PERSONAL_TOKEN"]
    api = HfApi(token=token)
    repo = "Tropic-AI/lumen-bench"
    info = api.dataset_info(repo, files_metadata=True)
    assert info.private, "Keep destination private until all release checks finish."
    remote = {r.rfilename: r for r in info.siblings}
    pending = []
    for row in selected:
        existing = remote.get(row["path"])
        data = (stage / row["path"]).read_bytes()
        expected = (row["sha256"] if existing and existing.lfs else
                    hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest())
        observed = (existing.lfs.sha256 if existing.lfs else existing.blob_id) if existing else None
        if expected != observed:
            pending.append(row)
    batches, batch, size = [], [], 0
    for row in pending:
        if batch and (len(batch) >= 400 or size + row["bytes"] > 32_000_000):
            batches.append(batch)
            batch, size = [], 0
        batch.append(row)
        size += row["bytes"]
    if batch:
        batches.append(batch)
    parent = info.sha
    commits = []
    try:
        for number, batch in enumerate(batches, 1):
            result = api.upload_folder(
                repo_id=repo, repo_type="dataset", folder_path=stage,
                allow_patterns=[r["path"] for r in batch], token=token,
                parent_commit=parent,
                commit_message=f"Audited release artifacts ({number}/{len(batches)})",
                run_as_future=False,
            )
            parent = result.oid
            commits.append(parent)
            save(ROOT / "analysis/verification/hf_upload_progress.json", {"repository": repo, "private": True,
                 "commit": parent, "batches_completed": number, "batches_total": len(batches), "commits": commits})
            print(json.dumps({"uploaded_batch": number, "batches": len(batches), "files": len(batch), "commit": parent}), flush=True)
    except Exception as exc:
        print(json.dumps({"upload_failed": type(exc).__name__,
                          "http_status": getattr(getattr(exc, "response", None), "status_code", None)}))
        raise SystemExit(1) from None
    receipt = {"repository": repo, "private": True, "parent_commit": info.sha,
               "commit": parent, "files": len(selected), "changed_files_uploaded": len(pending), "commits": commits,
               "mode": "static" if args.static else "full",
               "credential_values_printed": False}
    save(ROOT / "analysis/verification/hf_upload_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
