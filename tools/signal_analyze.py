"""
Did a convention actually emerge?

    python signal_analyze.py runs_signal/<timestamp>

Every measure here is computable from the log. The interesting one is
compositionality: accuracy above chance only shows that SOMETHING was
communicated, and a holistic code (one arbitrary label per whole object) can
score perfectly while being nothing like a language. Decomposition is the
harder and rarer property.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CHANCE = 0.25


def load(run_dir: Path) -> tuple[dict, list[dict]]:
    cfg, rounds = {}, []
    path = run_dir / "rounds.jsonl"
    if not path.exists():
        subs = sorted(d for d in run_dir.iterdir() if (d / "rounds.jsonl").exists())
        if not subs:
            raise SystemExit(f"No rounds.jsonl under {run_dir}")
        path = subs[-1] / "rounds.jsonl"
        print(f"(latest: {subs[-1]})\n")
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") == "round":
            rounds.append(rec)
        else:
            cfg.update(rec)
    return cfg, rounds


def learning_curve(rounds: list[dict], bins: int = 6) -> list[dict]:
    n = len(rounds)
    size = max(1, n // bins)
    out = []
    for i in range(0, n, size):
        chunk = rounds[i:i + size]
        if not chunk:
            continue
        acc = sum(r["correct"] for r in chunk) / len(chunk)
        out.append({"rounds": f"{chunk[0]['round']}-{chunk[-1]['round']}",
                    "n": len(chunk), "accuracy": round(acc, 3)})
    return out


def emergence(rounds: list[dict]) -> dict:
    """Beating chance is necessary but not sufficient. Compare the first and
    last thirds -- a convention that formed should show improvement, not just
    a lucky overall average."""
    n = len(rounds)
    third = max(1, n // 3)
    early = sum(r["correct"] for r in rounds[:third]) / third
    late = sum(r["correct"] for r in rounds[-third:]) / third
    overall = sum(r["correct"] for r in rounds) / n
    # binomial sd for the late window under the null
    sd = math.sqrt(CHANCE * (1 - CHANCE) / third)
    z = (late - CHANCE) / sd if sd else 0
    return {"overall": round(overall, 3), "early_third": round(early, 3),
            "late_third": round(late, 3), "improvement": round(late - early, 3),
            "late_z_vs_chance": round(z, 2),
            "verdict": ("convention formed" if z > 2 and late - early > 0.1 else
                        "above chance but not clearly learned" if z > 2 else
                        "no evidence of a shared code")}


def lexicon(rounds: list[dict], window: int | None = None) -> dict:
    """What does each symbol co-occur with? A symbol that has settled will be
    dominated by one attribute value."""
    use = rounds[-window:] if window else rounds
    by_symbol = defaultdict(lambda: {"color": Counter(), "shape": Counter(), "n": 0})
    for r in use:
        for s in r["symbols"]:
            by_symbol[s]["color"][r["target"]["color"]] += 1
            by_symbol[s]["shape"][r["target"]["shape"]] += 1
            by_symbol[s]["n"] += 1
    out = {}
    for sym, d in sorted(by_symbol.items(), key=lambda x: -x[1]["n"]):
        if not d["n"]:
            continue
        c_top, c_cnt = d["color"].most_common(1)[0]
        s_top, s_cnt = d["shape"].most_common(1)[0]
        out[sym] = {
            "uses": d["n"],
            "color": f"{c_top} ({c_cnt}/{d['n']} = {c_cnt/d['n']:.0%})",
            "shape": f"{s_top} ({s_cnt}/{d['n']} = {s_cnt/d['n']:.0%})",
            "purity": round(max(c_cnt, s_cnt) / d["n"], 2),
            "binds": ("color" if c_cnt > s_cnt else "shape" if s_cnt > c_cnt else "?"),
        }
    return out


def compositionality(rounds: list[dict], window: int = 30) -> dict:
    """Is the code built from parts?

    Holistic: each of the 16 objects gets its own arbitrary label, and knowing
    'ember-orb' tells you nothing about 'ember-spike'.
    Compositional: one symbol carries colour, another carries shape, and novel
    combinations work first time.

    Measured as: do messages for objects sharing a colour overlap more than
    messages for objects sharing nothing? Positive margin suggests parts."""
    use = rounds[-window:] if window else rounds
    msgs = defaultdict(list)
    for r in use:
        key = (r["target"]["color"], r["target"]["shape"])
        msgs[key].append(frozenset(r["symbols"]))

    def jaccard(a, b):
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b) if (a | b) else 0.0

    same_c, same_s, diff = [], [], []
    keys = list(msgs)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            pairs = [jaccard(m1, m2) for m1 in msgs[k1] for m2 in msgs[k2]]
            if not pairs:
                continue
            avg = sum(pairs) / len(pairs)
            if k1[0] == k2[0]:
                same_c.append(avg)
            elif k1[1] == k2[1]:
                same_s.append(avg)
            else:
                diff.append(avg)

    def m(x):
        return round(sum(x) / len(x), 3) if x else None

    mc, ms, md = m(same_c), m(same_s), m(diff)
    margin = None
    if md is not None and (mc is not None or ms is not None):
        best = max(v for v in (mc, ms) if v is not None)
        margin = round(best - md, 3)
    return {
        "overlap_same_color": mc, "overlap_same_shape": ms,
        "overlap_unrelated": md, "margin": margin,
        "reading": ("evidence of compositional structure" if margin and margin > 0.15
                    else "looks holistic — labels not built from parts"
                    if margin is not None else "not enough data"),
    }


def contamination(rounds: list[dict], cfg: dict) -> dict:
    """Are the models leaking natural language, or falling back on a real one?
    'They invented a code' and 'they annotated it in English' look identical in
    a transcript unless you check."""
    alphabet = set(cfg.get("alphabet", []))
    leaks, empty = [], 0
    for r in rounds:
        raw = (r.get("sender_raw") or "").lower()
        extra = [w for w in __import__("re").findall(r"[a-z]{2,}", raw)
                 if w not in alphabet]
        if extra:
            leaks.append({"round": r["round"], "extra": extra[:8],
                          "raw": r["sender_raw"][:120]})
        if not r["symbols"]:
            empty += 1
    return {"rounds_with_non_alphabet_text": len(leaks),
            "rounds_with_no_valid_symbol": empty,
            "examples": leaks[:5]}


def message_stats(rounds: list[dict]) -> dict:
    lens = Counter(len(r["symbols"]) for r in rounds)
    uniq = Counter(tuple(r["symbols"]) for r in rounds)
    return {"message_length_counts": dict(sorted(lens.items())),
            "distinct_messages": len(uniq),
            "most_common": [(" ".join(k) or "(none)", v) for k, v in uniq.most_common(5)]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    cfg, rounds = load(a.run_dir)
    if not rounds:
        raise SystemExit("No rounds logged.")

    report = {
        "rounds": len(rounds),
        "emergence": emergence(rounds),
        "learning_curve": learning_curve(rounds),
        "lexicon_last_30": lexicon(rounds, 30),
        "compositionality": compositionality(rounds),
        "contamination": contamination(rounds, cfg),
        "messages": message_stats(rounds),
    }
    if a.json:
        print(json.dumps(report, indent=2))
        return 0

    e = report["emergence"]
    print(f"\n{len(rounds)} rounds   chance = {CHANCE:.2f}\n" + "=" * 60)
    print(f"\naccuracy  overall {e['overall']}   early {e['early_third']} -> "
          f"late {e['late_third']}   (z vs chance = {e['late_z_vs_chance']})")
    print(f"VERDICT: {e['verdict']}")

    print("\nlearning curve")
    for b in report["learning_curve"]:
        bar = "█" * int(b["accuracy"] * 30)
        print(f"  r{b['rounds']:>9}  {b['accuracy']:.2f}  {bar}")

    print("\nlexicon (last 30 rounds)")
    for sym, d in report["lexicon_last_30"].items():
        print(f"  {sym:6s} x{d['uses']:<3d} binds {d['binds']:6s} "
              f"purity {d['purity']:.2f}   color={d['color']}  shape={d['shape']}")

    c = report["compositionality"]
    print(f"\ncompositionality  same-colour {c['overlap_same_color']}  "
          f"same-shape {c['overlap_same_shape']}  unrelated {c['overlap_unrelated']}"
          f"  margin {c['margin']}")
    print(f"  {c['reading']}")

    ct = report["contamination"]
    print(f"\ncontamination  {ct['rounds_with_non_alphabet_text']} rounds with "
          f"non-alphabet text, {ct['rounds_with_no_valid_symbol']} with no valid symbol")
    for ex in ct["examples"][:3]:
        print(f"    r{ex['round']}: {ex['extra']}  \"{ex['raw'][:70]}\"")

    ms = report["messages"]
    print(f"\nmessages  lengths {ms['message_length_counts']}  "
          f"distinct {ms['distinct_messages']}")
    print(f"  most common: {ms['most_common']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
