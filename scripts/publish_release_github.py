#!/usr/bin/env python3
"""Commit/push the audited isolated release to the authorized personal target.

Never imports workspace history, uses a work-account helper, changes global
Git settings, force-pushes, or changes visibility. Push requires a separate
clean staged-history credential scan after creating the commit.
"""
import argparse
import base64
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


class ExplicitPersonalToken(requests.auth.AuthBase):
    """Prevent .netrc/work-account credentials from overriding this token."""
    def __init__(self, token):
        self.token = token

    def __call__(self, request):
        request.headers["Authorization"] = "Bearer " + self.token
        return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    stage = args.directory.resolve()
    assert stage.parent == Path("/tmp") and stage.name.startswith("lumen-public-")
    manifest = json.loads((ROOT / "analysis/verification/release_candidate_manifest.json").read_text())
    allowed = {r["path"] for r in manifest}
    actual = {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file() and ".git" not in p.relative_to(stage).parts}
    assert actual == allowed, "Stage must contain exactly the screened publication manifest."
    for row in manifest:
        assert hashlib.sha256((stage / row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("GIT_TRACE") or key.startswith("GIT_CONFIG_") or key in {"GIT_ASKPASS", "GIT_CURL_VERBOSE", "GIT_DIR", "GIT_WORK_TREE"}:
            env.pop(key)
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/false")

    def git(*arguments, check=True):
        return subprocess.run(["git", *arguments], cwd=stage, env=env, capture_output=True, text=True, check=check)

    if args.commit:
        assert git("rev-parse", "--verify", "HEAD", check=False).returncode != 0, "Only create a fresh release history."
        git("config", "user.name", "TropicAI-Research")
        git("config", "user.email", "noreply@users.noreply.github.com")
        git("config", "core.hooksPath", "/dev/null")
        # This isolated tree was checked against the exact screened manifest
        # above. Include archived evidence even when the working project's
        # ignore rules normally exclude incremental response logs.
        git("add", "--force", "--all", ".")
        indexed = set(git("ls-files", "-z").stdout.rstrip("\0").split("\0"))
        assert indexed == allowed, "The publication index must include every audited evidence file."
        git("-c", "commit.gpgsign=false", "commit", "-m", "Release corrected Lumen-bench with auditable source records")
        head = git("rev-parse", "HEAD").stdout.strip()
        save(ROOT / "analysis/verification/github_release_commit.json", {"commit": head, "fresh_history": True, "files": len(manifest)})
        print(json.dumps({"commit": head, "files": len(manifest), "uploaded": False}), flush=True)
    if not args.push:
        return
    head = git("rev-parse", "HEAD").stdout.strip()
    checks = json.loads((ROOT / "analysis/verification/staged_history_security.json").read_text())
    assert checks["commit"] == head and checks["passed"] is True
    token = env_values()["GITHUB_PERSONAL_TOKEN2"]
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json"}
    session = requests.Session()
    session.auth = ExplicitPersonalToken(token)
    api = "https://api.github.com/repos/TropicAI-Research/lumen"
    info = session.get(api, headers=headers, timeout=30)
    assert info.status_code == 200
    data = info.json()
    assert data["private"] and data["permissions"]["push"]
    remote = session.get(api + "/commits", headers=headers, timeout=30)
    if remote.status_code == 200:
        assert remote.json()[0]["sha"] == head, "Remote is not the expected empty repository; do not overwrite."
    else:
        assert remote.status_code == 409, "Unexpected repository state."
    encoded = base64.b64encode(("x-access-token:" + token).encode()).decode()
    env.update(GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
               GIT_CONFIG_VALUE_0="Authorization: Basic " + encoded,
               GIT_CONFIG_KEY_1="credential.helper", GIT_CONFIG_VALUE_1="")
    result = git("push", "https://github.com/TropicAI-Research/lumen.git", "HEAD:refs/heads/main", check=False)
    if result.returncode:
        print(json.dumps({"push_failed": True, "exit_code": result.returncode}), flush=True)
        raise SystemExit(1)
    verify = session.get(api + "/commits/main", headers=headers, timeout=30)
    assert verify.status_code == 200 and verify.json()["sha"] == head
    receipt = {"repository": "TropicAI-Research/lumen", "commit": head, "private": True,
               "credential_variable": "GITHUB_PERSONAL_TOKEN2", "global_authentication_modified": False}
    save(ROOT / "analysis/verification/github_upload_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
