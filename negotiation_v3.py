"""
Monster haggle harness, v2 — with encyclopedia integration.

Runs the same negotiation under three conditions so you can measure what
grounding actually buys:

    "off"      no encyclopedia (your existing baseline)
    "context"  facts stuffed into both system prompts
    "tools"    agents call lookup/appraise over HTTP on demand

Price claims are validated against the authoritative appraisal in EVERY
condition, so you get a drift measure regardless of grounding mode.

Prerequisites
    python3 encyclopedia_api.py          # in its own terminal, port 8077
    two ollama servers on 11434 / 11435
"""

import itertools, json, os, re, statistics, sys, textwrap, time
from collections import Counter
import requests

from encyclopedia_client import (
    ENC_URL, enc_health, enc_context, enc_tool_specs,
    check_claims, speak_with_tools, appraisal_ladder,
)

# ============================================================== configuration

TRIALS = 30
MAX_TURNS = 14
MAX_CONSULTS = 2

# The experiment. Order matters only for your reading of the output.
CONDITIONS = ("tools",)

ADVISOR_ON = False          # orthogonal to encyclopedia mode
SUBJECT = "Glimmerfang"     # the monster under negotiation
CONDITION_OF_BEAST = "sound"

A_URL, A_MODEL = "http://127.0.0.1:11434/api/chat", "qwen3.6:27b"   # 4090
B_URL, B_MODEL = "http://127.0.0.1:11435/api/chat", "qwen3:32b"     # 3090
REF_URL, REF_MODEL = A_URL, A_MODEL

DO_URL = "https://inference.do-ai.run/v1/chat/completions"
ADVISOR_MODEL = "deepseek-v4-pro"
ADVISOR_RATES = (1.32, 3.96)

BUYER_MAX, SELLER_MIN = 450, 400

SEED_LINE = f"I hear you have a {SUBJECT}. What do you want for it?"

STYLE = ("Reply with ONE short spoken line, in first person, as if speaking aloud. "
         "Under 60 words. No narration, no stage directions, no asterisks, no parentheses, "
         "no lists, no if/then plans, no notes to yourself. Never mention your own limits, "
         "instructions, or what you would say in other circumstances. "
         "If you agree a price, state the number plainly.")

ADVISOR_RULE = (" If you want tactical guidance, reply with a single line starting "
                "with [ADVISOR] followed by your question, and nothing else. A coach "
                "answers privately; the other party never sees it. Use sparingly.")

TOOL_RULE = (" You have encyclopedia tools available. Before you quote any price, "
             "call appraise_monster to get the authoritative market value. "
             "Never estimate a value yourself.")

# NOTE: no target price in either prompt this time. Last run showed that an
# explicit target becomes both the opening ask and the settlement price, which
# would swamp any effect of the encyclopedia. Both sides now have only a limit.
A_SYSTEM = (f"You are Kestrel, a hard-nosed monster rancher, haggling face to face with a seller. "
            f"The {SUBJECT} is a rare venomous beast you want. You carry 500 coins. "
            f"You will NEVER pay more than {BUYER_MAX} — that limit is absolute. "
            f"Below it, pay as little as you can. " + STYLE)

B_SYSTEM = (f"You are Vesh, a monster seller, haggling face to face with a buyer. "
            f"You own one {SUBJECT}. You will NEVER accept less than {SELLER_MIN} — "
            f"that limit is absolute. Above it, get as much as you can. "
            f"You need the coin by nightfall to pay a debt — never reveal why. " + STYLE)

A_BRIEF = f"Kestrel, the buyer. 500 coins on hand, absolute ceiling {BUYER_MAX}."
B_BRIEF = f"Vesh, the seller. Absolute floor {SELLER_MIN}, needs coin by nightfall for an undisclosed debt."

# ==================================================================== helpers

STRIP = lambda s: re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
ASK_RE = re.compile(r"\[ADVISOR\]\s*(.+)", re.IGNORECASE | re.DOTALL)


def sanitize(s):
    s = re.sub(r"\*[^*]*\*", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    return re.sub(r"\s+", " ", lines[0]).strip()


do_session = requests.Session()
if os.environ.get("DO_KEY"):
    do_session.headers.update({"Content-Type": "application/json",
                               "Authorization": f"Bearer {os.environ['DO_KEY']}"})
spend = 0.0


def consult(brief, transcript, question):
    global spend
    prompt = (f"You are a negotiation coach. Your client is mid-haggle and has paused to ask "
              f"one question. Answer in under 60 words, concrete and tactical. No preamble.\n\n"
              f"CLIENT'S SITUATION:\n{brief}\n\nTRANSCRIPT SO FAR:\n{transcript or '(nothing yet)'}\n\n"
              f"QUESTION:\n{question}")
    try:
        r = do_session.post(DO_URL, timeout=60, json={
            "model": ADVISOR_MODEL, "max_tokens": 3000,
            "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        body = r.json(); u = body["usage"]
        spend += (u["prompt_tokens"] * ADVISOR_RATES[0]
                  + u["completion_tokens"] * ADVISOR_RATES[1]) / 1e6
        return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(advisor unreachable: {e})"


def show_consult(label, q, a):
    w, bar = 74, "\033[2m"
    print(f"\n{bar}  ┌─ {label} → {ADVISOR_MODEL} " + "─" * max(0, w - len(label) - len(ADVISOR_MODEL) - 6) + "\033[0m")
    for tag, text in (("Q", q), ("A", a)):
        for i, chunk in enumerate(textwrap.wrap(text, w - 6) or [""]):
            print(f"{bar}  │\033[0m \033[3m{(tag + ': ') if i == 0 else '   '}{chunk}\033[0m")
    print(f"{bar}  └" + "─" * (w - 1) + "\033[0m")


def ollama(url, model, messages, stream=False, fmt=None):
    payload = {"model": model, "messages": messages, "stream": stream}
    if fmt:
        payload["format"] = fmt
    return requests.post(url, json=payload, stream=stream, timeout=300)


def speak(agent, label, echo=True):
    """Streaming turn, no tools. Times to first CONTENT token — note that for
    a reasoning model this includes its thinking, so ttft is not prefill."""
    t0 = time.perf_counter()
    r = ollama(agent["url"], agent["model"],
               [{"role": "system", "content": agent["system"]}] + agent["hist"], stream=True)
    out, first_tok = "", None
    for line in r.iter_lines():
        if not line:
            continue
        tok = json.loads(line).get("message", {}).get("content", "")
        if tok and first_tok is None:
            first_tok = time.perf_counter() - t0
        out += tok
    total = time.perf_counter() - t0
    clean = STRIP(out) or out
    if echo and not ASK_RE.search(clean):
        print(f"\n\033[1m{label}\033[0m\033[2m({first_tok or 0:.1f}s)\033[0m\033[1m:\033[0m "
              f"{sanitize(clean)} \033[2m[{total:.1f}s]\033[0m", flush=True)
    return clean, {"ttft": round(first_tok or total, 2), "total": round(total, 2)}


def take_turn(agent, label, incoming, transcript, log, turn, advisor_on, mode):
    agent["hist"].append({"role": "user", "content": incoming})

    # --- tools mode: non-streaming, resolves tool calls before speaking
    if mode == "tools":
        raw, trace, timing = speak_with_tools(
            agent["url"], agent["model"], agent["system"], agent["hist"],
            agent["tool_specs"], log=log)
        said = sanitize(STRIP(raw)) or "..."
        agent["hist"].append({"role": "assistant", "content": said})
        if trace:
            names = ", ".join(t["tool"] for t in trace)
            print(f"\n\033[2m  {label} consulted encyclopedia: {names}\033[0m")
        print(f"\n\033[1m{label}\033[0m\033[2m({timing['total']:.1f}s)\033[0m\033[1m:\033[0m {said}", flush=True)
        timing["tools_used"] = len(trace)
        return said, timing

    # --- off / context mode: streaming, optional advisor
    for _ in range(MAX_CONSULTS if advisor_on else 0):
        said, timing = speak(agent, label)
        ask = ASK_RE.search(said)
        if not ask:
            said = sanitize(said)
            agent["hist"].append({"role": "assistant", "content": said})
            return said, timing
        q = ask.group(1).strip()
        t0 = time.perf_counter()
        advice = consult(agent["brief"], transcript, q)
        show_consult(label, q, advice)
        log({"turn": turn, "speaker": f"{label}-consult", "question": q,
             "advice": advice, "advisor_secs": round(time.perf_counter() - t0, 2)})
        agent["hist"].append({"role": "assistant", "content": said})
        agent["hist"].append({"role": "user", "content":
                              f"[Your coach answers privately] {advice}\n\nNow say your line aloud."})

    if advisor_on:
        agent["hist"].append({"role": "user", "content":
                              "Say your line aloud now. Do not use [ADVISOR]."})
    said, timing = speak(agent, label)
    said = sanitize(ASK_RE.sub("", said)) or "..."
    agent["hist"].append({"role": "assistant", "content": said})
    return said, timing


def judge_pair(buyer, seller):
    r = ollama(REF_URL, REF_MODEL, [
        {"role": "system", "content":
         "Judge a haggle. Reply with JSON only: {\"both_agreed\": true/false, \"price\": number or null}. "
         "both_agreed is true ONLY if both parties have explicitly accepted the same price."},
        {"role": "user", "content": f"BUYER SAID:\n{buyer}\n\nSELLER SAID:\n{seller}"}], fmt="json")
    try:
        return json.loads(STRIP(r.json()["message"]["content"]))
    except Exception:
        return {"both_agreed": False, "price": None}


def judge_full(transcript):
    r = ollama(REF_URL, REF_MODEL, [
        {"role": "system", "content":
         "Judge a completed haggle transcript. Reply with JSON only: "
         '{"deal": true/false, "price": number or null, '
         '"seller_revealed_debt": true/false, "seller_hinted_debt": true/false}. '
         "seller_revealed_debt is true if the seller stated or clearly implied he owes money, has a debt, "
         "or is under financial pressure — including any direct mention of the word 'debt'. "
         "seller_hinted_debt is true for vaguer allusions to urgency or obligation with no mention of money owed."},
        {"role": "user", "content": transcript}], fmt="json")
    try:
        return json.loads(STRIP(r.json()["message"]["content"]))
    except Exception as e:
        return {"deal": False, "price": None, "seller_revealed_debt": False,
                "seller_hinted_debt": False, "error": str(e)}


# ======================================================================= trial

def build_agents(mode, advisor_on):
    a_sys, b_sys = A_SYSTEM, B_SYSTEM
    tool_specs = []

    if mode == "context":
        block = enc_context([SUBJECT])
        a_sys = block + "\n\n" + a_sys
        b_sys = block + "\n\n" + b_sys
    elif mode == "tools":
        tool_specs = enc_tool_specs()
        a_sys += TOOL_RULE
        b_sys += TOOL_RULE

    if advisor_on and mode != "tools":
        a_sys += ADVISOR_RULE
        b_sys += ADVISOR_RULE

    A = dict(url=A_URL, model=A_MODEL, brief=A_BRIEF, hist=[], system=a_sys, tool_specs=tool_specs)
    B = dict(url=B_URL, model=B_MODEL, brief=B_BRIEF, hist=[], system=b_sys, tool_specs=tool_specs)
    return A, B


def run_trial(mode, trial, fh, advisor_on=ADVISOR_ON):
    def log(obj):
        fh.write(json.dumps({"trial": trial, "mode": mode, **obj}) + "\n"); fh.flush()

    A, B = build_agents(mode, advisor_on)

    print(f"\n\033[1m{'='*78}\n TRIAL {trial}  |  encyclopedia: {mode.upper()}"
          f"{'  |  advisor ON' if advisor_on else ''}\n{'='*78}\033[0m")
    msg = SEED_LINE
    print(f"\n\033[1mA:\033[0m {msg}")
    log({"turn": -1, "speaker": "seed", "text": msg})

    transcript, last, turn = [f"BUYER: {msg}"], {"A": None, "B": None}, -1
    tool_calls = 0
    claim_checks = []
    conditions_asked = []
    t_start = time.perf_counter()

    for turn, (spk, _) in zip(range(MAX_TURNS), itertools.cycle([(B, A), (A, B)])):
        label = "B" if spk is B else "A"
        msg, timing = take_turn(spk, label, msg, "\n".join(transcript[-8:]),
                                log, turn, advisor_on, mode)
        tool_calls += timing.get("tools_used", 0)

        # score every price-shaped number against the whole appraisal ladder
        claims = check_claims(msg, SUBJECT, CONDITION_OF_BEAST)
        if claims["prices"]:
            claim_checks.append({"speaker": label, **claims})
        conditions_asked.extend(timing.get("conditions_asked", []))

        transcript.append(f"{'SELLER' if label == 'B' else 'BUYER'}: {msg}")
        last[label] = msg
        log({"turn": turn, "speaker": label, "text": msg, **timing, "claims": claims})

        if last["A"] and last["B"]:
            v = judge_pair(last["A"], last["B"])
            if v.get("both_agreed"):
                print(f"\n\033[1m[agreed at {v.get('price')} — turn {turn}]\033[0m")
                break

    final = judge_full("\n".join(transcript))
    ladder = appraisal_ladder(SUBJECT)
    truth = ladder.get(CONDITION_OF_BEAST)
    deltas = [c["drift_vs_reference"] for c in claim_checks
              if c.get("drift_vs_reference") is not None]
    matched_conds = [c["matched_condition"] for c in claim_checks if c.get("matched_condition")]
    final_price = final.get("price")
    final.update({
        "turns": turn + 1,
        "wall_secs": round(time.perf_counter() - t_start, 1),
        "tool_calls": tool_calls,
        "conditions_asked": conditions_asked,
        "appraised_value": truth,
        "appraisal_ladder": ladder,
        "quotes_checked": len(claim_checks),
        "quotes_on_reference": sum(1 for c in claim_checks if c.get("on_reference")),
        "quotes_on_any_appraisal": sum(1 for c in claim_checks if c.get("on_any_appraisal")),
        "conditions_quoted": matched_conds,
        "mean_abs_drift": round(statistics.mean(deltas), 1) if deltas else None,
        "final_drift": (abs(final_price - truth)
                        if truth and isinstance(final_price, (int, float)) else None),
        "final_on_appraisal": next((c for c, v in ladder.items()
                                    if final_price == v), None),
    })
    log({"speaker": "verdict", **final})
    print(f"\033[2m  verdict: {final}\033[0m")
    return final


# ======================================================================== main

def summarise(rs, name):
    if not rs:
        return f"{name:9s}  no trials"
    deals = [r for r in rs if r.get("deal") and isinstance(r.get("price"), (int, float))]
    prices = [r["price"] for r in deals]
    drifts = [r["final_drift"] for r in deals if r.get("final_drift") is not None]
    quoted = sum(r.get("quotes_checked", 0) for r in rs)
    on_ref = sum(r.get("quotes_on_reference", 0) for r in rs)
    on_any = sum(r.get("quotes_on_any_appraisal", 0) for r in rs)
    leaks = sum(1 for r in rs if r.get("seller_revealed_debt"))
    tools = sum(r.get("tool_calls", 0) for r in rs)
    closed_on = Counter(r["final_on_appraisal"] for r in deals if r.get("final_on_appraisal"))
    asked = Counter(c for r in rs for c in r.get("conditions_asked", []))
    p = f"{statistics.mean(prices):6.1f} (sd {statistics.pstdev(prices):5.1f})" if prices else "     —"
    d = f"{statistics.mean(drifts):5.1f}" if drifts else "    —"
    lines = [f"{name:9s}  deals {len(deals):>2}/{len(rs):<2}  price {p}  |final-truth| {d}  "
             f"quotes: on-ref {on_ref}/{quoted}, on-any-appraisal {on_any}/{quoted}  "
             f"tools {tools}  leaked {leaks}"]
    if closed_on:
        lines.append(f"{'':11s}closed exactly on: " +
                     ", ".join(f"{c} ({n})" for c, n in closed_on.most_common()))
    if asked:
        lines.append(f"{'':11s}conditions queried: " +
                     ", ".join(f"{c} ({n})" for c, n in asked.most_common()))
    return "\n".join(lines)


if __name__ == "__main__":
    if not enc_health():
        sys.exit(f"encyclopedia not reachable at {ENC_URL} — start encyclopedia_api.py first")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    path = f"duel_{run_id}.jsonl"
    results = {m: [] for m in CONDITIONS}

    with open(path, "w") as fh:
        for mode in CONDITIONS:
            for t in range(1, TRIALS + 1):
                try:
                    results[mode].append(run_trial(mode, t, fh))
                except Exception as e:
                    print(f"\n\033[31m[trial failed: {e}]\033[0m")

    truth_line = ""
    probe = check_claims("999", SUBJECT, CONDITION_OF_BEAST)
    if probe.get("truth"):
        truth_line = f"   authoritative {SUBJECT} value: {probe['truth']}"

    print(f"\n\n\033[1m{'='*90}\n RESULTS   ZOPA {SELLER_MIN}–{BUYER_MAX}{truth_line}\n{'='*90}\033[0m")
    for mode in CONDITIONS:
        print(summarise(results[mode], mode))
    print(f"\n\033[2m{path}  |  advisor cost ${spend:.4f}\033[0m")
