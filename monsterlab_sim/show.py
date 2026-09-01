"""
Read a run as a story instead of as JSON.

    python show.py runs/20260811-133300
    python show.py runs/ --agent iris          # one agent only
    python show.py runs/ --day 2               # one day
    python show.py runs/ --only reasoning      # just the thinking
    python show.py runs/ --full                # don't truncate

Interleaves reasoning, actions, and messages in the order they happened, which
is the view you need to answer the question the aggregate stats can't: is this
agent modelling the other agent at all?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

C = {"reasoning": "\033[90m", "action": "\033[36m", "message": "\033[33m",
     "no_tool_call": "\033[31m", "error": "\033[31m", "day": "\033[1m", "off": "\033[0m"}


def load(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        subs = sorted(d for d in run_dir.iterdir() if (d / "events.jsonl").exists())
        if not subs:
            raise SystemExit(f"No events.jsonl under {run_dir}")
        path = subs[-1] / "events.jsonl"
        print(f"(latest run: {subs[-1]})\n")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def clip(s: str, n: int, full: bool) -> str:
    s = " ".join(str(s).split())
    return s if full or len(s) <= n else s[:n] + "…"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--agent")
    p.add_argument("--day", type=int)
    p.add_argument("--only", choices=["reasoning", "action", "message"])
    p.add_argument("--full", action="store_true")
    p.add_argument("--no-color", action="store_true")
    a = p.parse_args()

    col = {k: ("" if a.no_color else v) for k, v in C.items()}
    day = None
    shown = 0

    for e in load(a.run_dir):
        t = e.get("type")
        if a.agent and e.get("agent") != a.agent:
            continue
        if a.day and e.get("day") != a.day:
            continue

        if t == "day_start" and not a.only:
            day = e["day"]
            print(f"\n{col['day']}{'='*70}\nDAY {day}{col['off']}")
            for o in (e.get("market") or {}).get("orders", []):
                if not o["filled_by"]:
                    req = " + ".join(
                        f"{r['rarity']} {r.get('element') or r['kind']}"
                        for r in o.get("requires", [])
                    ) or "?"
                    tag = "BUNDLE " if o.get("bundle") else ""
                    print(f"  {tag}order {o['order_id']}: {req} → {o['pays']}c")
            for name, st in (e.get("agents") or {}).items():
                inv = ", ".join(
                    f"#{c['id']} {c.get('species', c['kind'])} {c.get('rarity','')}"
                    for c in st.get("inventory", [])
                ) or "empty"
                print(f"  {name}: {st['energy']}e {st['coins']}c | {inv}")
            continue

        if a.only and t != a.only:
            continue
        shown += 1

        if t == "reasoning":
            print(f"\n{col['reasoning']}  {e['agent']} thinks: "
                  f"{clip(e['content'], 500, a.full)}{col['off']}")
        elif t == "action":
            r = e.get("result") or {}
            res = r.get("payload", {}).get("result") if "payload" in r else r
            if isinstance(res, dict) and res.get("species"):
                summary = (f"{res['species']} {res.get('rarity','')} "
                           f"(#{res.get('id')}, worth {res.get('appraisal','?')}c)")
            elif isinstance(res, dict) and res.get("caught") is False:
                summary = "got away"
            else:
                summary = clip(json.dumps(res, ensure_ascii=False), 160, a.full)
            args = ", ".join(f"{k}={v}" for k, v in (e.get("args") or {}).items())
            print(f"{col['action']}  {e['agent']} → {e['tool']}({args}): {summary}{col['off']}")
        elif t == "message":
            print(f"{col['message']}  {e['agent']} ⇒ {e['to']}: "
                  f"\"{clip(e['content'], 400, a.full)}\"{col['off']}")
            if e.get("reasoning_before"):
                print(f"{col['reasoning']}      (was thinking: "
                      f"{clip(e['reasoning_before'], 200, a.full)}){col['off']}")
        elif t == "no_tool_call":
            print(f"{col['no_tool_call']}  {e['agent']} [no tool call] "
                  f"{clip(e.get('content',''), 200, a.full)}{col['off']}")
        elif t == "error":
            print(f"{col['error']}  {e['agent']} ERROR {e.get('error')}{col['off']}")

    if not shown:
        print("Nothing matched those filters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
