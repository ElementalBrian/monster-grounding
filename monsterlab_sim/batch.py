"""
Run the same configuration N times and aggregate. This is the step that turns
"it happened once" into "it happens X% of the time", which is the difference
between an anecdote and a result.

    python batch.py --trials 10 --days 4
    python batch.py --trials 6 --config cross_model.json --label cross
    python batch.py --report runs/batch-20260811-1500      # re-aggregate later

Each trial gets a fresh world (--reset) and its own run directory. Nothing is
shared between trials except the configuration, so they are independent samples.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import analyze


def summarise(run_dir: Path) -> dict:
    """One row per trial. Keep this to things that are computable, not judged."""
    events = analyze.load(run_dir)
    h = analyze.health(events)
    tr = analyze.trade_activity(events)
    cb = analyze.combat_activity(events)
    pr = analyze.promises_vs_transfers(events)
    act = analyze.activity(events)
    lb = next((e.get("scored_leaderboard") for e in reversed(events)
               if e.get("type") == "run_end" and e.get("scored_leaderboard")),
              next((e["leaderboard"] for e in reversed(events)
                    if e.get("type") == "day_end" and not e.get("winddown")), []))
    fills = sum(c.get("fill_order", 0) for c in act["tool_use"].values())

    return {
        "run": run_dir.name,
        "messages": sum(act["messages_sent"].values()),
        "trades_proposed": tr["proposed"],
        "trades_accepted": tr["accepted"],
        "trades_failed": tr["failed"],
        "trades_unresolvable": tr["unresolvable"],
        "battles_issued": cb["issued"],
        "battles_fought": cb["accepted"],
        "battles_declined": cb["declined"],
        "unbacked_roster_claims": len(cb["roster_claims_unbacked"]),
        "promises": pr["promises"],
        "promises_kept": pr["honoured"],
        "misleading_intent": len(analyze.reasoning_vs_message(events)),
        "orders_filled": fills,
        "winner": lb[0]["agent_id"] if lb else None,
        "spread": (lb[0]["net_worth"] - lb[-1]["net_worth"]) if len(lb) > 1 else None,
        "net_worth": {a["agent_id"]: a["net_worth"] for a in lb},
        "minutes": h["wall_clock_min"],
        "problems": h["problems"],
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)

    def rate(key):
        return round(sum(1 for r in rows if r.get(key)) / n, 3) if n else None

    def stat(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        return {
            "mean": round(statistics.mean(vals), 2),
            "sd": round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals),
        }

    winners = {}
    for r in rows:
        if r.get("winner"):
            winners[r["winner"]] = winners.get(r["winner"], 0) + 1

    return {
        "trials": n,
        "rates": {
            "any_message": rate("messages"),
            "any_trade_proposed": rate("trades_proposed"),
            "any_trade_accepted": rate("trades_accepted"),
            "any_misleading_intent": rate("misleading_intent"),
            "any_battle_issued": rate("battles_issued"),
            "any_battle_fought": rate("battles_fought"),
            "any_unbacked_roster_claim": rate("unbacked_roster_claims"),
        },
        "per_trial": {k: stat(k) for k in
                      ("messages", "trades_proposed", "trades_accepted",
                       "battles_issued", "battles_fought", "battles_declined",
                       "orders_filled", "spread", "minutes")},
        "promise_keep_rate": (
            round(sum(r["promises_kept"] for r in rows) /
                  sum(r["promises"] for r in rows), 3)
            if sum(r["promises"] for r in rows) else None
        ),
        "wins": winners,
        "trials_with_problems": sum(1 for r in rows if r["problems"]),
    }


def report(batch_dir: Path) -> dict:
    all_rows = [summarise(d) for d in sorted(batch_dir.iterdir())
                if (d / "events.jsonl").exists()]
    # An interrupted trial has no run_end. Including it silently drags every
    # rate toward zero -- a Ctrl+C should not look like a behavioural result.
    rows = [r for r in all_rows if not any("no run_end" in p for p in r["problems"])]
    dropped = [r["run"] for r in all_rows if r not in rows]
    agg = aggregate(rows)
    agg["incomplete_excluded"] = dropped
    out = {"aggregate": agg, "trials": rows}
    (batch_dir / "summary.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'='*66}\n{agg['trials']} complete trials — {batch_dir}\n{'='*66}")
    if agg.get("incomplete_excluded"):
        print(f"  (excluded {len(agg['incomplete_excluded'])} incomplete: "
              f"{', '.join(agg['incomplete_excluded'])})")
    print("\nhow often did it happen at all")
    for k, v in agg["rates"].items():
        bar = "█" * int((v or 0) * 20)
        print(f"  {k:24s} {v if v is not None else '-':>6}  {bar}")
    print("\nper trial (mean ± sd, range)")
    for k, s in agg["per_trial"].items():
        if s:
            print(f"  {k:18s} {s['mean']:>7} ± {s['sd']:<6} [{s['min']} … {s['max']}]")
    if agg["promise_keep_rate"] is not None:
        print(f"\npromises kept overall: {agg['promise_keep_rate']}")
    print(f"wins: {agg['wins']}")
    if agg["trials_with_problems"]:
        print(f"\n[!] {agg['trials_with_problems']}/{agg['trials']} trials had health problems")
        for r in rows:
            if r["problems"]:
                print(f"    {r['run']}: {'; '.join(r['problems'])}")
    print(f"\nwrote {batch_dir / 'summary.json'}\n")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--days", type=int, default=4)
    p.add_argument("--budget", type=int, default=8)
    p.add_argument("--comms", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--config", default="agents.json")
    p.add_argument("--label", default="batch")
    p.add_argument("--lab", default="http://localhost:8000")
    p.add_argument("--ollama", default="http://localhost:11434")
    p.add_argument("--report", type=Path, help="re-aggregate an existing batch and exit")
    p.add_argument("--extra", nargs=argparse.REMAINDER,
                   help="anything else to pass through to runner.py")
    a = p.parse_args()

    if a.report:
        report(a.report)
        return 0

    batch_dir = Path(f"runs/{a.label}-{time.strftime('%Y%m%d-%H%M%S')}")
    batch_dir.mkdir(parents=True)
    print(f"{a.trials} trials → {batch_dir}\n")

    for i in range(1, a.trials + 1):
        run_dir = batch_dir / f"trial-{i:02d}"
        cmd = [sys.executable, "runner.py", "--config", a.config, "--days", str(a.days),
               "--budget", str(a.budget), "--comms", str(a.comms), "--rounds", str(a.rounds), "--reset",
               "--lab", a.lab, "--ollama", a.ollama, "--run-dir", str(run_dir)] + (a.extra or [])
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        mins = (time.time() - t0) / 60
        if r.returncode != 0:
            print(f"  trial {i:02d} FAILED after {mins:.1f}m: {r.stderr.strip()[-200:]}")
            continue
        try:
            s = summarise(run_dir)
            print(f"  trial {i:02d}  {mins:4.1f}m  msgs={s['messages']:2d} "
                  f"prop={s['trades_proposed']} acc={s['trades_accepted']} "
                  f"btl={s['battles_issued']}/{s['battles_fought']} "
                  f"filled={s['orders_filled']:2d}  winner={s['winner']}")
        except Exception as e:
            print(f"  trial {i:02d} ran but could not be summarised: {e!r}")

    report(batch_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
