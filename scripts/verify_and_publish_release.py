#!/usr/bin/env python3
"""Verify exact remote contents, then optionally make the two personal targets public.

An explicit --make-public performs the visibility changes authorized by the
repository owner. No arXiv submission or work-account authentication.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_language_repairs import env_values, save
from scripts.publish_release_github import ExplicitPersonalToken


class NoAccountAuthentication(requests.auth.AuthBase):
    def __call__(self, request):
        request.headers.pop("Authorization", None)
        return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--make-public", action="store_true")
    args = parser.parse_args()
    stage = args.directory.resolve()
    assert stage.parent == Path("/tmp") and stage.name.startswith("lumen-public-")
    manifest = json.loads((ROOT / "analysis/verification/release_candidate_manifest.json").read_text())
    security = json.loads((ROOT / "analysis/verification/staged_history_security.json").read_text())
    assert security["passed"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=stage, text=True).strip()
    assert head == security["commit"]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=stage)
    credentials = env_values()
    github = requests.Session()
    github.auth = ExplicitPersonalToken(credentials["GITHUB_PERSONAL_TOKEN2"])
    gh_api = "https://api.github.com/repos/TropicAI-Research/lumen"
    commit = github.get(gh_api + "/commits/main", timeout=30)
    assert commit.status_code == 200 and commit.json()["sha"] == head
    tree = github.get(gh_api + "/git/trees/" + head, params={"recursive": 1}, timeout=60)
    assert tree.status_code == 200 and not tree.json()["truncated"]
    gh_files = {r["path"]: r for r in tree.json()["tree"] if r["type"] == "blob"}
    assert set(gh_files) == {r["path"] for r in manifest}
    os.environ["HF_HOME"] = "/tmp/lumen-hf-upload-cache"
    from huggingface_hub import HfApi
    hf = HfApi(token=credentials["HF_PERSONAL_TOKEN"])
    hf_repo = "Tropic-AI/lumen-bench"
    hf_info = hf.dataset_info(hf_repo, files_metadata=True)
    hf_files = {r.rfilename: r for r in hf_info.siblings}
    assert set(hf_files) - {".gitattributes"} == {r["path"] for r in manifest}
    for row in manifest:
        data = (stage / row["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert gh_files[row["path"]]["sha"] == blob and gh_files[row["path"]]["size"] == len(data), row["path"]
        remote = hf_files[row["path"]]
        assert remote.size == len(data), row["path"]
        assert (remote.lfs.sha256 == row["sha256"] if remote.lfs else remote.blob_id == blob), row["path"]
    print(json.dumps({"both_remote_trees_match": True, "files": len(manifest), "github_commit": head, "hf_commit": hf_info.sha}), flush=True)
    if not args.make_public:
        return
    changed = github.patch(gh_api, json={"private": False}, timeout=30)
    assert changed.status_code == 200 and changed.json()["private"] is False
    hf.update_repo_settings(repo_id=hf_repo, repo_type="dataset", private=False, token=credentials["HF_PERSONAL_TOKEN"])
    anonymous = requests.Session()
    anonymous.auth = NoAccountAuthentication()
    gh_public = anonymous.get(gh_api, timeout=30)
    hf_public = anonymous.get("https://huggingface.co/api/datasets/" + hf_repo, timeout=30)
    assert gh_public.status_code == hf_public.status_code == 200
    assert not gh_public.json()["private"] and not hf_public.json()["private"]
    checks = []
    for row in manifest:
        if row["path"] not in {"release/lumen-bench.pdf", "release/lumen-bench-arxiv.tar.gz", "croissant.json", "data/tables/parallel_inputs.csv"}:
            continue
        for name, base in [("github", "https://raw.githubusercontent.com/TropicAI-Research/lumen/main/"),
                           ("huggingface", "https://huggingface.co/datasets/Tropic-AI/lumen-bench/resolve/main/")]:
            response = anonymous.get(base + row["path"], timeout=60)
            assert response.status_code == 200 and hashlib.sha256(response.content).hexdigest() == row["sha256"]
            checks.append({"service": name, "path": row["path"], "sha256": row["sha256"]})
    receipt = {"github": "https://github.com/TropicAI-Research/lumen", "huggingface": "https://huggingface.co/datasets/Tropic-AI/lumen-bench",
        "github_commit": head, "hf_commit": hf_info.sha, "both_public": True,
        "verified_files_on_each_service": len(manifest), "anonymous_downloads": checks,
        "work_account_authentication_modified": False, "arxiv_submitted": False}
    save(ROOT / "analysis/verification/publication_receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
