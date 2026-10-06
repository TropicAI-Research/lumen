#!/usr/bin/env python3
"""Provision the explicitly authorized, budget-limited repair key.

Credential values are written only to the git-ignored .env via apply_patch.
Neither API key nor administrative credential is logged.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Dry run: use --execute to create the authorized US$12 key.")
    path = ROOT / ".env"
    content = path.read_text()
    values = dict(line.split("=", 1) for line in content.splitlines()
                  if "=" in line and not line.lstrip().startswith("#"))
    if values.get("OPENROUTER_REPAIR_KEY_HASH"):
        print("Task key already provisioned. No duplicate key created.")
        return
    admin = values["OPENROUTER_API_KEY"].strip()
    headers = {"Authorization": "Bearer " + admin}
    check = requests.get("https://openrouter.ai/api/v1/key", headers=headers, timeout=30)
    check.raise_for_status()
    if not check.json()["data"].get("is_management_key"):
        raise SystemExit("Current key is not administrative; no key-management action taken.")
    expires = (datetime.now(timezone.utc) + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    name = "lumen-language-repair-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    response = requests.post("https://openrouter.ai/api/v1/keys", headers=headers,
        json={"name": name, "limit": 12, "expires_at": expires}, timeout=30)
    if not response.ok:
        try:
            error = response.json().get("error", {})
            print(json.dumps({"status": response.status_code, "error_message": error.get("message"),
                              "error_metadata": error.get("metadata")}))
        except (ValueError, AttributeError):
            print("Key provisioning failed without a JSON diagnostic.")
    response.raise_for_status()
    data = response.json()
    key = data["key"]
    key_hash = data["data"]["hash"]
    patch = ("*** Begin Patch\n*** Update File: .env\n@@\n-OPENROUTER_API_KEY=" + admin
        + "\n+OPENROUTER_API_KEY=" + key + "\n+OPENROUTER_MANAGEMENT_API_KEY=" + admin
        + "\n+OPENROUTER_REPAIR_KEY_HASH=" + key_hash + "\n*** End Patch\n")
    result = subprocess.run(["apply_patch"], input=patch, text=True, capture_output=True, cwd=ROOT)
    if result.returncode:
        # Do not log the patch or subprocess diagnostics (they may contain keys).
        requests.patch("https://openrouter.ai/api/v1/keys/" + key_hash, headers=headers,
                       json={"disabled": True}, timeout=30).raise_for_status()
        raise SystemExit("Could not save credential; disabled the newly created key.")
    os.chmod(path, 0o600)
    metadata = {"name": name, "limit_usd": 12, "limit_reset": None,
                "expires_at": expires, "user_authorized": True,
                "scope": "Lumen-bench translation repair, affected-case reruns and semantic judging"}
    out = ROOT / "data/language_repair/budget.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
