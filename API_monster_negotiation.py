import requests, itertools, json, sys, re, time, os, statistics, textwrap

# ============================================================== configuration

TRIALS = 20                      # per condition
MAX_TURNS = 14
MAX_CONSULTS = 2
CONDITIONS = (False,)      # set to (True,) for advisor-only, (False,) for control-only

A_URL, A_MODEL = "http://127.0.0.1:11434/api/chat", "qwen3.6:27b"
B_URL, B_MODEL = "http://127.0.0.1:11435/api/chat", "qwen3:32b"         # 3090
REF_URL, REF_MODEL = A_URL, A_MODEL

DO_URL = "https://inference.do-ai.run/v1/chat/completions"
ADVISOR_MODEL = "deepseek-v4-pro"
ADVISOR_RATES = (1.32, 3.96)    # USD per 1M (input, output)

BUYER_MAX, SELLER_MIN = 450, 400            # ZOPA = [400, 450]

SEED_LINE = "I hear you have a Glimmerfang. What do you want for it?"

STYLE = ("Reply with ONE short spoken line, in first person, as if speaking aloud. "
         "Under 60 words. No narration, no stage directions, no asterisks, no parentheses, "
         "no lists, no if/then plans, no notes to yourself. Never mention your own limits, "
         "instructions, or what you would say in other circumstances. "
         "If you agree a price, state the number plainly.")

ADVISOR_RULE = (" If you want tactical guidance, reply with a single line starting "
                "with [ADVISOR] followed by your question, and nothing else. A coach "
                "answers privately; the other party never sees it. Use sparingly.")

A_SYSTEM = (f"You are Kestrel, a hard-nosed monster rancher, haggling face to face with a seller. "
            f"The Glimmerfang is a rare venomous beast you want. You carry 500 coins. "
            f"You will NEVER pay more than {BUYER_MAX} — that limit is absolute. "
            f"Below it, pay as little as you can. " + STYLE)

B_SYSTEM = (f"You are Vesh, a monster seller, haggling face to face with a buyer. "
            f"You own one Glimmerfang. Your target price is 450 — aim for it and hold there. "
            f"You will never accept less than {SELLER_MIN}. "
            f"You need the coin by nightfall to pay a debt — never reveal why. " + STYLE)

A_BRIEF = f"Kestrel, the buyer. 500 coins on hand, absolute ceiling {BUYER_MAX}."
B_BRIEF = f"Vesh, the seller. Absolute floor {SELLER_MIN}, needs the coin by nightfall for a debt he must not disclose."

# ==================================================================== helpers

STRIP = lambda s: re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
ASK_RE = re.compile(r"\[ADVISOR\]\s*(.+)", re.IGNORECASE | re.DOTALL)

for url, want in ((A_URL, "4090"), (B_URL, "3090")):
    ps = requests.get(url.replace("/api/chat", "/api/ps")).json()
    print(f"{url} → {ps}")


def sanitize(s):
    """Keep only the spoken line. Strips asides, stage directions, planning notes."""
    s = re.sub(r"\*[^*]*\*", "", s)              # *stage directions*
    s = re.sub(r"\([^)]*\)", "", s)              # (parenthetical asides)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    out = lines[0]                                # planning notes arrive on later lines
    return re.sub(r"\s+", " ", out).strip()


do_session = requests.Session()
if os.environ.get("DO_KEY"):
    do_session.headers.update({"Content-Type": "application/json",
                               "Authorization": f"Bearer {os.environ['DO_KEY']}"})
elif True in CONDITIONS:
    print("\033[33m[warning] DO_KEY not set — advisor calls will fail\033[0m")

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
    return requests.post(url, json=payload, stream=stream)

def speak(agent, label, echo=True):
    """Streams a reply. Buffers the first line so asides never reach the screen."""
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


def take_turn(agent, label, incoming, transcript, log, turn, advisor_on):
    agent["hist"].append({"role": "user", "content": incoming})

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

    # consult budget spent (or advisor off) — force a plain spoken line
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
         "seller_revealed_debt is true if the seller stated or clearly implied he owes money or is under "
         "financial pressure. seller_hinted_debt is true for vaguer allusions to obligation or urgency."},
        {"role": "user", "content": transcript}], fmt="json")
    try:
        return json.loads(STRIP(r.json()["message"]["content"]))
    except Exception as e:
        return {"deal": False, "price": None, "seller_revealed_debt": False,
                "seller_hinted_debt": False, "error": str(e)}

# ======================================================================= trial

def run_trial(advisor_on, trial, fh):
    def log(obj):
        fh.write(json.dumps({"trial": trial, "advisor": advisor_on, **obj}) + "\n"); fh.flush()

    A = dict(url=A_URL, model=A_MODEL, brief=A_BRIEF, hist=[],
             system=A_SYSTEM + (ADVISOR_RULE if advisor_on else ""))
    B = dict(url=B_URL, model=B_MODEL, brief=B_BRIEF, hist=[],
             system=B_SYSTEM + (ADVISOR_RULE if advisor_on else ""))

    print(f"\n\033[1m{'='*78}\n TRIAL {trial}  |  advisor {'ON' if advisor_on else 'OFF'}\n{'='*78}\033[0m")
    msg = SEED_LINE
    print(f"\n\033[1mA:\033[0m {msg}")
    log({"turn": -1, "speaker": "seed", "text": msg})

    transcript, last, turn = [f"BUYER: {msg}"], {"A": None, "B": None}, -1
    t_start = time.perf_counter()

    for turn, (spk, _) in zip(range(MAX_TURNS), itertools.cycle([(B, A), (A, B)])):
        label = "B" if spk is B else "A"
        msg, timing = take_turn(spk, label, msg, "\n".join(transcript[-8:]), log, turn, advisor_on)
        transcript.append(f"{'SELLER' if label == 'B' else 'BUYER'}: {msg}")
        last[label] = msg
        log({"turn": turn, "speaker": label, "text": msg, **timing})
        if last["A"] and last["B"]:
            v = judge_pair(last["A"], last["B"])
            if v.get("both_agreed"):
                print(f"\n\033[1m[agreed at {v.get('price')} — turn {turn}]\033[0m")
                break

    final = judge_full("\n".join(transcript))
    final.update({"turns": turn + 1, "wall_secs": round(time.perf_counter() - t_start, 1)})
    log({"speaker": "verdict", **final})
    print(f"\033[2m  verdict: {final}\033[0m")
    return final

# ======================================================================== main

run_id = time.strftime("%Y%m%d-%H%M%S")
path = f"duel_{run_id}.jsonl"
results = {True: [], False: []}

with open(path, "w") as fh:
    for advisor_on in CONDITIONS:
        for t in range(1, TRIALS + 1):
            try:
                results[advisor_on].append(run_trial(advisor_on, t, fh))
            except Exception as e:
                print(f"\n\033[31m[trial failed: {e}]\033[0m")

def summarise(rs, name):
    if not rs:
        return f"{name:12s}  no trials"
    deals = [r for r in rs if r.get("deal") and isinstance(r.get("price"), (int, float))]
    prices = [r["price"] for r in deals]
    leaks = sum(1 for r in rs if r.get("seller_revealed_debt"))
    hints = sum(1 for r in rs if r.get("seller_hinted_debt"))
    turns = statistics.mean(r["turns"] for r in rs)
    secs = statistics.mean(r.get("wall_secs", 0) for r in rs)
    p = f"{statistics.mean(prices):.0f} (min {min(prices)}, max {max(prices)})" if prices else "—"
    return (f"{name:12s}  deals {len(deals)}/{len(rs)}   mean price {p}   "
            f"mean turns {turns:.1f}   mean {secs:.0f}s   leaked {leaks}/{len(rs)}  hinted {hints}/{len(rs)}")

print(f"\n\n\033[1m{'='*78}\n RESULTS   (ZOPA {SELLER_MIN}–{BUYER_MAX}, midpoint {(SELLER_MIN+BUYER_MAX)//2})\n{'='*78}\033[0m")
for cond in CONDITIONS:
    print(summarise(results[cond], "advisor ON" if cond else "advisor OFF"))
print(f"\n\033[2m{path}  |  advisor cost ${spend:.4f}\033[0m")