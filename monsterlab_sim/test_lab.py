"""
Sanity tests. Run with: python test_lab.py

Checks the things that would silently ruin an experiment: distributions that
don't match the declared table, non-determinism, and receipts that can't
actually be verified.
"""

import collections
import os
import secrets
import tempfile

import fair

os.environ.setdefault("MONSTERLAB_DB", tempfile.mktemp(suffix=".db"))
os.environ.setdefault("MONSTERLAB_KEY", tempfile.mktemp(suffix=".key"))

PASS, FAIL = "  PASS", "  FAIL"
failures = []


def check(name, cond, detail=""):
    print((PASS if cond else FAIL) + f"  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def test_determinism():
    seed = secrets.token_bytes(32)
    a = fair.roll_monster(seed, "alice", 7)
    b = fair.roll_monster(seed, "alice", 7)
    c = fair.roll_monster(seed, "alice", 8)
    d = fair.roll_monster(seed, "bob", 7)
    check("same inputs give same monster", a == b)
    check("different nonce gives different monster", a != c)
    check("different agent gives different monster", a != d)


def test_rarity_distribution(n=200_000):
    seed = secrets.token_bytes(32)
    counts = collections.Counter(
        fair.roll_monster(seed, "a", i)["rarity"] for i in range(n)
    )
    expected = {k: w / sum(w for _, w in fair.RARITY) for k, w in fair.RARITY}
    worst = 0.0
    for rarity, p in expected.items():
        actual = counts[rarity] / n
        worst = max(worst, abs(actual - p) / p)
    check(
        "rarity distribution matches declared table",
        worst < 0.05,
        f"max relative error {worst:.3%} over {n:,} rolls",
    )


def test_catch_rate():
    seed = secrets.token_bytes(32)
    rolls = [fair.roll_monster(seed, "a", i) for i in range(50_000)]
    by_rarity = collections.defaultdict(lambda: [0, 0])
    for r in rolls:
        slot = by_rarity[r["rarity"]]
        slot[1] += 1
        if r["caught"]:
            slot[0] += 1
    ordered = [by_rarity[r][0] / by_rarity[r][1] for r in fair.RARITY_ORDER if by_rarity[r][1]]
    check(
        "rarer monsters escape more often",
        all(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)),
        " > ".join(f"{x:.0%}" for x in ordered),
    )


def test_breeding_improves_stats():
    seed = secrets.token_bytes(32)
    parent = {
        "kind": "monster", "caught": True, "species": "Mossbuck", "rarity": "common",
        "element": "verdant", "hp": 40, "attack": 40, "speed": 40,
        "fertility": 50, "generation": 0, "id": 1,
    }
    kids = [fair.breed(seed, "a", i, parent, {**parent, "id": 2}) for i in range(5000)]
    viable = [k for k in kids if k.get("caught")]
    mean_hp = sum(k["hp"] for k in viable) / len(viable)
    upgrades = sum(1 for k in viable if k["rarity"] != "common") / len(viable)
    check("breeding is usually viable", 0.80 < len(viable) / len(kids) < 0.92,
          f"{len(viable)/len(kids):.1%}")
    check("offspring stats drift upward from midparent", mean_hp > 40,
          f"mean hp {mean_hp:.1f} vs parents 40")
    check("rarity upgrades happen but stay uncommon", 0.15 < upgrades < 0.30,
          f"{upgrades:.1%}")


def test_commit_reveal():
    seed = secrets.token_bytes(32)
    commit = fair.commit_for(seed)
    import verify
    check("correct seed matches commitment", verify.check_commit(seed.hex(), commit))
    check(
        "wrong seed fails commitment",
        not verify.check_commit(secrets.token_bytes(32).hex(), commit),
    )


def test_end_to_end():
    """Boot the API in-process, run actions, verify the receipts."""
    from fastapi.testclient import TestClient
    import server
    import verify

    client = TestClient(server.app)
    info = client.get("/").json()
    pubkey = info["public_key"]

    reg = client.post("/register", json={"agent_id": "alice"}).json()
    h = {"X-Agent-Id": "alice", "X-Agent-Key": reg["api_key"]}

    r1 = client.post("/catch", headers=h).json()
    check("catch returns a signed receipt", "signature" in r1 and "payload" in r1)
    check("signature verifies", verify.check_signature(r1, pubkey))

    tampered = {**r1, "payload": {**r1["payload"], "result": {**r1["payload"]["result"], "rarity": "legendary"}}}
    check("tampered receipt fails verification", not verify.check_signature(tampered, pubkey))

    client.post("/fish", headers=h)
    me = client.get("/me", headers=h).json()
    check("energy was debited", me["energy"] == server.START_ENERGY - 15,
          f"{me['energy']} left")

    reveal = client.post("/round/reveal").json()
    check("revealed seed matches published commit",
          verify.check_commit(reveal["seed"], reveal["commit"]))

    log = client.get(f"/round/log/{reveal['revealed_round']}").json()
    audit = verify.audit_round(log, pubkey)
    check("whole round recomputes from the seed", audit["ok"], str(audit["verified"]))

    # insufficient funds path
    for _ in range(20):
        client.post("/catch", headers=h)
    broke = client.post("/catch", headers=h).json()
    check("runs out of energy eventually", broke.get("detail", "").startswith("Insufficient"))


if __name__ == "__main__":
    print("\nMonsterLab test suite\n" + "-" * 60)
    for fn in [
        test_determinism,
        test_rarity_distribution,
        test_catch_rate,
        test_breeding_improves_stats,
        test_commit_reveal,
        test_end_to_end,
    ]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "-" * 60)
    print(f"{'ALL PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}\n")
    raise SystemExit(1 if failures else 0)
