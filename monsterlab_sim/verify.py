"""
Standalone verifier. Depends only on `fair.py` and the server's public key --
never on the server itself. Give this to your agents as a tool.

Two independent checks:

  1. signature   -- was this receipt really issued by the server?
  2. recompute   -- given the revealed seed, does the outcome actually follow?

Check 1 works during a round. Check 2 only works after /round/reveal, which
is the whole point of commit-reveal: the server publishes a hash up front and
cannot retroactively choose a seed that flatters anyone.

Usage:
    python verify.py receipt.json --pubkey <hex>
    python verify.py --audit-round round1.json     # verify a whole round
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

import fair


def check_signature(receipt: dict, pubkey_hex: str) -> bool:
    pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
    try:
        pub.verify(bytes.fromhex(receipt["signature"]), fair.canonical(receipt["payload"]))
        return True
    except InvalidSignature:
        return False


def strip_derived(result: dict) -> dict:
    """Server adds id/appraisal after rolling; those aren't part of the roll."""
    return {k: v for k, v in result.items() if k not in ("id", "appraisal")}


def recompute(payload: dict, seed_hex: str, parents: dict | None = None) -> dict:
    """`parents` also carries {"home_element": ...} for catch recomputation --
    the receipt records it, so pass payload.get("home_element") through."""
    seed = bytes.fromhex(seed_hex)
    agent, nonce, action = payload["agent"], payload["nonce"], payload["action"]

    if action == "catch":
        # Hunting rights override the home element for that catch, so the
        # receipt records which territory was actually hunted.
        expected = fair.roll_monster(
            seed, agent, nonce,
            home_element=payload.get("hunting_element") or payload.get("home_element"))
    elif action == "fish":
        expected = fair.roll_fish(seed, agent, nonce)
    elif action == "breed":
        if not parents:
            return {"ok": None, "reason": "breed needs parent traits to recompute"}
        expected = fair.breed(seed, agent, nonce, parents["a"], parents["b"])
    else:
        return {"ok": None, "reason": f"unknown action {action}"}

    actual = strip_derived(payload["result"])
    expected = strip_derived(expected)
    return {"ok": expected == actual, "expected": expected, "actual": actual}


def check_commit(seed_hex: str, commit_hex: str) -> bool:
    return hashlib.sha256(bytes.fromhex(seed_hex)).hexdigest() == commit_hex


def audit_round(round_blob: dict, pubkey_hex: str | None = None) -> dict:
    """Verify an entire round dump from GET /round/log/{id}."""
    seed = round_blob.get("seed")
    if not seed:
        return {"ok": False, "reason": "round not yet revealed"}
    if not check_commit(seed, round_blob["commit"]):
        return {"ok": False, "reason": "SEED DOES NOT MATCH COMMITMENT -- server cheated"}

    results = {"catch": [0, 0], "fish": [0, 0], "breed": [0, 0], "skipped": 0}
    bad = []
    for entry in round_blob["entries"]:
        # breed needs parent traits and battle needs both rosters, neither of
        # which the round log carries -- they're signed but not recomputable here.
        if entry["action"] in ("breed", "battle"):
            results["skipped"] += 1
            continue
        r = recompute(entry, seed)
        slot = results[entry["action"]]
        slot[1] += 1
        if r["ok"]:
            slot[0] += 1
        else:
            bad.append({"agent": entry["agent"], "nonce": entry["nonce"], **r})

    return {
        "ok": not bad,
        "commit_valid": True,
        "verified": {k: f"{v[0]}/{v[1]}" for k, v in results.items() if k != "skipped"},
        "skipped_breeds": results["skipped"],
        "mismatches": bad,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="receipt JSON, or round log with --audit-round")
    ap.add_argument("--pubkey", help="server public key (hex)")
    ap.add_argument("--seed", help="revealed round seed (hex)")
    ap.add_argument("--audit-round", action="store_true")
    args = ap.parse_args()

    with open(args.path) as f:
        blob = json.load(f)

    if args.audit_round:
        print(json.dumps(audit_round(blob, args.pubkey), indent=2))
        return 0

    out = {}
    if args.pubkey:
        out["signature_valid"] = check_signature(blob, args.pubkey)
    if args.seed:
        out["recomputed"] = recompute(blob["payload"], args.seed)
    if not out:
        print("Nothing to check: pass --pubkey and/or --seed.", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    return 0 if all(v is not False for v in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
