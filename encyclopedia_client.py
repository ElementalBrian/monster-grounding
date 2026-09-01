"""
Client-side glue: lets Ollama agents reach the encyclopedia over HTTP.

Three integration levels, runnable as experimental conditions:

    "off"      no encyclopedia
    "context"  facts stuffed into the system prompt
    "tools"    agents call lookup/appraise over HTTP on demand

v2 measurement changes
----------------------
* parse_numbers() replaces the phrase lookup — handles "four hundred
  twenty-seven", "five-oh-three", "three-eighty" and friends.
* check_claims() now scores against the FULL appraisal ladder, not just
  `sound`. A seller quoting the pristine value is computing correctly and
  arguing self-servingly; that is a different thing from fabricating, and
  the old scoring collapsed the two. Reported separately now.
* speak_with_tools() returns which conditions the agent asked about, so
  you can see what inputs it chose rather than only what number it said.
"""

import json
import time

import requests

from numparse import parse_numbers

ENC_URL = "http://127.0.0.1:8077"
_session = requests.Session()

CONDITIONS_ALL = ("pristine", "sound", "scarred", "ailing")
_ladder_cache = {}


# ================================================================== retrieval

def enc_health():
    try:
        return _session.get(f"{ENC_URL}/health", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def enc_context(names=None, formula=True):
    try:
        params = {"formula": str(formula).lower()}
        if names:
            params["names"] = ",".join(names)
        r = _session.get(f"{ENC_URL}/context", params=params, timeout=10)
        r.raise_for_status()
        return r.json()["text"]
    except requests.RequestException as e:
        print(f"[encyclopedia unreachable: {e}]")
        return ""


def enc_tool_specs():
    try:
        r = _session.get(f"{ENC_URL}/tools", timeout=10)
        r.raise_for_status()
        return r.json()["tools"]
    except requests.RequestException:
        return []


def enc_call(name, args):
    try:
        r = _session.post(f"{ENC_URL}/call", json={"name": name, "args": args or {}}, timeout=15)
        return r.json()
    except requests.RequestException as e:
        return {"error": f"encyclopedia unreachable: {e}"}


def appraisal_ladder(monster):
    """{condition: value} for every condition. Cached — it never changes."""
    if monster in _ladder_cache:
        return _ladder_cache[monster]
    ladder = {}
    for c in CONDITIONS_ALL:
        r = enc_call("appraise_monster", {"name": monster, "condition": c})
        if "value" in r:
            ladder[c] = r["value"]
    if ladder:
        _ladder_cache[monster] = ladder
    return ladder


# =================================================================== validator

def check_claims(text, monster, reference="sound", tolerance=0):
    """Score the numbers in one line against the whole appraisal ladder.

    Returns:
      prices              every number spoken
      truth               the reference-condition value (what the scenario says)
      on_reference        a quote equals the reference appraisal
      on_any_appraisal    a quote equals SOME valid appraisal
      matched_condition   which one, if any
      drift_vs_reference  |nearest quote - reference value|

    The on_reference / on_any_appraisal split is the point: an agent quoting
    the pristine value has done the arithmetic correctly and chosen a
    self-serving input. That is not the same failure as inventing a number,
    and scoring only against `sound` made them indistinguishable.
    """
    prices = parse_numbers(text)
    ladder = appraisal_ladder(monster)
    truth = ladder.get(reference)

    out = {"prices": prices, "truth": truth, "on_reference": False,
           "on_any_appraisal": False, "matched_condition": None,
           "drift_vs_reference": None}
    if not prices or not ladder:
        return out

    for cond in CONDITIONS_ALL:
        val = ladder.get(cond)
        if val is None:
            continue
        if any(abs(p - val) <= tolerance for p in prices):
            out["on_any_appraisal"] = True
            out["matched_condition"] = cond
            out["on_reference"] = (cond == reference)
            if cond == reference:
                break

    if truth is not None:
        out["drift_vs_reference"] = min(abs(p - truth) for p in prices)
    return out


# ============================================================== tool-call loop

def speak_with_tools(url, model, system, hist, tool_specs, max_tool_rounds=4, log=None):
    """Non-streaming turn that resolves tool calls before returning.

    Returns (text, trace, timing). timing includes `conditions_asked` — the
    condition arguments the agent passed to appraise_monster, which is the
    record of what inputs it chose.
    """
    t0 = time.perf_counter()
    messages = [{"role": "system", "content": system}] + list(hist)
    trace = []

    def finish(text):
        conds = [t["args"].get("condition", "sound") for t in trace
                 if t["tool"] == "appraise_monster"]
        return text, trace, {
            "total": round(time.perf_counter() - t0, 2),
            "tools_used": len(trace),
            "conditions_asked": conds,
        }

    for _ in range(max_tool_rounds):
        r = _session.post(url, timeout=300, json={
            "model": model, "messages": messages, "tools": tool_specs, "stream": False})
        msg = r.json().get("message", {}) or {}
        calls = msg.get("tool_calls") or []

        if not calls:
            return finish(msg.get("content", "") or "")

        messages.append(msg)
        for tc in calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = enc_call(name, args)
            trace.append({"tool": name, "args": args, "result": result})
            if log:
                log({"speaker": "tool", "tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result)})

    messages.append({"role": "user", "content": "Answer now without calling any more tools."})
    r = _session.post(url, timeout=300, json={"model": model, "messages": messages, "stream": False})
    return finish((r.json().get("message", {}) or {}).get("content", "") or "")


if __name__ == "__main__":
    print("encyclopedia reachable:", enc_health())
    print("ladder:", appraisal_ladder("Glimmerfang"))
    print()
    for line in [
        "Four hundred fifty. Take it or leave it.",
        "Five hundred three coins, and not a jot less.",
        "Four hundred twenty-seven is the market value.",
        "Appraised at two-fifty-one, but I'm holding at four hundred.",
        "Three-fifty, final offer.",
    ]:
        c = check_claims(line, "Glimmerfang")
        print(f"{str(c['prices']):12s} ref={c['on_reference']!s:5s} any={c['on_any_appraisal']!s:5s} "
              f"cond={str(c['matched_condition']):9s} drift={c['drift_vs_reference']}  <- {line[:46]}")
