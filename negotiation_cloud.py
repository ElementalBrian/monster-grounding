"""
Monster haggle harness — cloud edition (DigitalOcean inference API).

Same experiment as the local version, but both agents and the referee run on
a hosted model instead of two local Ollama servers.

    export DO_KEY=...
    python3 encyclopedia_api.py          # still local, port 8077
    python3 negotiation_cloud.py

Differences from the Ollama version, all in the transport layer:
  * OpenAI-compatible schema: tool arguments arrive as JSON *strings*, and
    tool results must carry `tool_call_id` (Ollama used `tool_name`).
  * Streaming is SSE (`data: {...}`), not newline-delimited JSON.
  * JSON mode is `response_format`, not `format`.
  * Retries with backoff, because a network hop can fail in ways a local
    socket doesn't.

CAVEAT worth stating in any write-up: the local runs used two *different*
models (qwen3.6:27b vs qwen3:32b). If you point both agents at the same
cloud model you are also removing that asymmetry, so results are not
directly comparable to the local arms unless you keep two distinct models.
"""

import atexit, itertools, json, os, random, re, signal, statistics, sys, time
from collections import Counter, defaultdict

import requests

from encyclopedia_client import (
    ENC_URL, enc_health, enc_context, enc_tool_specs, enc_call,
    check_claims, appraisal_ladder,
)

# ============================================================== configuration

TRIALS = 30
MAX_TURNS = 14
CONDITIONS = ("off", "context", "tools")
# CONDITIONS = ("tools",)

SUBJECT = "Glimmerfang"
CONDITION_OF_BEAST = "sound"
BUYER_MAX, SELLER_MIN = 450, 400

DO_URL = "https://inference.do-ai.run/v1/chat/completions"

# Available: kimi-k2.5, kimi-k2.6, kimi-k3, qwen3.8-max, qwen3.5-397b-a17b,
#            deepseek-v4-pro, deepseek-4-flash, glm-5.2
# Keeping A and B distinct preserves the asymmetry the local runs had.
MODEL_A   = "kimi-k2.6"     # buyer  (was qwen3.6:27b)
MODEL_B   = "kimi-k2.6"     # seller (was qwen3:32b)
MODEL_REF = "kimi-k2.6"     # judge  (was the buyer's model)

# Kimi K2.6 is a reasoning model: it emits `reasoning_content` alongside
# `content`, and BOTH draw on max_tokens. A budget sized for the visible line
# alone gets consumed by thinking and returns content=null with
# finish_reason="length". Hence the headroom.
MAX_TOKENS = 20000               # spoken line + invisible reasoning
JUDGE_TOKENS = 8000         # judges reason too, then emit small JSON

# This backend rejects response_format json_object for kimi-k2.6. The judge
# prompts already demand JSON and the parser strips reasoning, so it isn't
# needed — flip to True only if you switch to a model that supports it.
JSON_MODE = False
TEMPERATURE = 1.0
REQUEST_TIMEOUT = 120
MAX_RETRIES = 8
STREAM = True               # live output for non-tool turns

# USD per 1M tokens (input, output).
RATES = defaultdict(lambda: (0.0, 0.0), {
    "kimi-k2.6": (0.76, 3.20),          # confirmed from the pricing page
    "kimi-k2.5": (0.76, 3.20),          # unverified
    "kimi-k3": (0.76, 3.20),            # unverified
    "deepseek-v4-pro": (1.32, 3.96),    # unverified
})

# Everything the run writes goes here. Override with LOG_DIR=/some/path.
LOG_DIR = os.environ.get("LOG_DIR", "logs")
# Append-only ledger across every run of this script.
USAGE_LOG = os.path.join(LOG_DIR, "token_usage.log")
# Rewritten after every trial so a hard kill still leaves a record. Deleted on
# a clean finish; if it exists at startup, the previous run died and its
# tokens get folded into the ledger before this run begins.
CHECKPOINT = os.path.join(LOG_DIR, "token_usage.current.json")

SEED_LINE = f"I hear you have a {SUBJECT}. What do you want for it?"

STYLE = ("Reply with ONE short spoken line, in first person, as if speaking aloud. "
         "Under 60 words. No narration, no stage directions, no asterisks, no parentheses, "
         "no lists, no if/then plans, no notes to yourself. Never mention your own limits, "
         "instructions, or what you would say in other circumstances. "
         "If you agree a price, state the number plainly.")

TOOL_RULE = (" You have encyclopedia tools available. Before you quote any price, "
             "call appraise_monster to get the authoritative market value. "
             "Never estimate a value yourself.")

A_SYSTEM = (f"You are Kestrel, a hard-nosed monster rancher, haggling face to face with a seller. "
            f"The {SUBJECT} is a rare venomous beast you want. You carry 500 coins. "
            f"You will NEVER pay more than {BUYER_MAX} — that limit is absolute. "
            f"Below it, pay as little as you can. " + STYLE)

B_SYSTEM = (f"You are Vesh, a monster seller, haggling face to face with a buyer. "
            f"You own one {SUBJECT}. You will NEVER accept less than {SELLER_MIN} — "
            f"that limit is absolute. Above it, get as much as you can. "
            f"You need the coin by nightfall to pay a debt — never reveal why. " + STYLE)

# ==================================================================== transport

# Checked lazily so offline modes (--recover) work without a key.
if not os.environ.get("DO_KEY") and "--recover" not in sys.argv:
    sys.exit("DO_KEY not set — export your DigitalOcean inference key first")

_session = requests.Session()
_session.headers.update({
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ.get('DO_KEY', '')}",
})

os.makedirs(LOG_DIR, exist_ok=True)

USAGE = defaultdict(lambda: {"prompt": 0, "completion": 0, "calls": 0})


SHOW_CALL_USAGE = False       # set True to echo tokens after every single call


def _account(model, usage, label=""):
    if not usage:
        return
    p = usage.get("prompt_tokens", 0)
    c = usage.get("completion_tokens", 0)
    u = USAGE[model]
    u["prompt"] += p
    u["completion"] += c
    u["calls"] += 1
    if SHOW_CALL_USAGE:
        print(f"\033[2m    {label or model}  prompt: {p:<7} completion: {c:<7} "
              f"total: {p + c}\033[0m", flush=True)


def cost_of(model, prompt_tok, completion_tok):
    rin, rout = RATES[model]
    return (prompt_tok * rin + completion_tok * rout) / 1e6


def usage_block(usage_map, indent="  "):
    """Render a {model: {prompt, completion, calls}} map as aligned lines."""
    lines, tp, tc, tcost = [], 0, 0, 0.0
    for m, u in sorted(usage_map.items()):
        cost = cost_of(m, u["prompt"], u["completion"])
        tp += u["prompt"]; tc += u["completion"]; tcost += cost
        lines.append(f"{indent}{m:<20} {u['calls']:>5} calls   "
                     f"prompt: {u['prompt']:>9,}   completion: {u['completion']:>8,}   "
                     f"total: {u['prompt'] + u['completion']:>9,}   ${cost:.4f}")
    lines.append(f"{indent}{'ALL':<20} {sum(u['calls'] for u in usage_map.values()):>5} calls   "
                 f"prompt: {tp:>9,}   completion: {tc:>8,}   "
                 f"total: {tp + tc:>9,}   ${tcost:.4f}")
    return "\n".join(lines)


def snapshot():
    return {m: dict(u) for m, u in USAGE.items()}


# ---------------------------------------------------------------- durability
#
# Tokens are spent the moment a request returns, so accounting must survive
# whatever happens next. Three layers:
#   1. CHECKPOINT rewritten after every trial  -> survives SIGKILL / power loss
#   2. finalise() in a finally + atexit + SIGTERM -> survives crash / Ctrl-C
#   3. per-trial usage in the JSONL             -> survives everything, and lets
#                                                  you recompute from transcripts
_RUN = {"id": None, "path": None, "start": None, "trials": 0, "written": False}


def write_checkpoint():
    try:
        tmp = CHECKPOINT + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"run_id": _RUN["id"], "transcript": _RUN["path"],
                       "trials": _RUN["trials"], "usage": snapshot(),
                       "cost": round(total_spend(), 6),
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
        os.replace(tmp, CHECKPOINT)          # atomic
    except OSError as e:
        print(f"\033[33m  [checkpoint write failed: {e}]\033[0m", flush=True)


def recover_checkpoint():
    """Fold an abandoned run's tokens into the ledger before starting a new one."""
    if not os.path.exists(CHECKPOINT):
        return
    try:
        with open(CHECKPOINT) as f:
            old = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    u = old.get("usage", {})
    tp = sum(x["prompt"] for x in u.values())
    tc = sum(x["completion"] for x in u.values())
    print(f"\033[33mprevious run {old.get('run_id')} did not finish — "
          f"recovering {tp + tc:,} tokens (${old.get('cost', 0):.4f})\033[0m")
    with open(USAGE_LOG, "a") as lf:
        lf.write(f"\n{'='*100}\n")
        lf.write(f"run {old.get('run_id')}   {old.get('updated')}   RECOVERED (killed)   "
                 f"{old.get('trials', 0)} trials completed\n")
        lf.write(f"transcript: {old.get('transcript')}\n")
        lf.write(usage_block(u) + "\n")
        lf.write(f"  prompt: {tp}  completion: {tc}  total: {tp + tc}  "
                 f"cost: ${old.get('cost', 0):.4f}\n")
    os.replace(CHECKPOINT, CHECKPOINT + ".recovered")


def finalise(status="complete"):
    """Write this run's block to the ledger. Safe to call more than once."""
    if _RUN["written"] or not USAGE:
        return
    _RUN["written"] = True
    tp = sum(u["prompt"] for u in USAGE.values())
    tc = sum(u["completion"] for u in USAGE.values())
    mins = (time.perf_counter() - _RUN["start"]) / 60 if _RUN["start"] else 0
    try:
        with open(USAGE_LOG, "a") as lf:
            lf.write(f"\n{'='*100}\n")
            lf.write(f"run {_RUN['id']}   {time.strftime('%Y-%m-%d %H:%M:%S')}   "
                     f"{mins:.1f} min   {_RUN['trials']} trials   {status.upper()}   "
                     f"conditions={','.join(CONDITIONS)}   "
                     f"A={MODEL_A} B={MODEL_B} REF={MODEL_REF}\n")
            lf.write(f"transcript: {_RUN['path']}\n")
            lf.write(usage_block(USAGE) + "\n")
            lf.write(f"  prompt: {tp}  completion: {tc}  total: {tp + tc}  "
                     f"cost: ${total_spend():.4f}\n")
    except OSError as e:
        print(f"\033[31m  [LEDGER WRITE FAILED: {e}]\033[0m")
        print(f"\033[31m  tokens this run: {tp} in, {tc} out, ${total_spend():.4f}\033[0m")
        return
    if status == "complete":
        try:
            os.remove(CHECKPOINT)
        except OSError:
            pass
    print(f"\033[2m  ledger updated ({status}): {tp + tc:,} tokens, "
          f"${total_spend():.4f}\033[0m", flush=True)


def lifetime():
    """(runs, tokens, usd) across every run in the ledger."""
    tok, usd, runs = 0, 0.0, 0
    try:
        for line in open(USAGE_LOG):
            if line.strip().startswith("prompt:") and "cost: $" in line:
                runs += 1
                tok += int(line.split("total:")[1].split()[0])
                usd += float(line.split("cost: $")[1].strip())
    except OSError:
        pass
    return runs, tok, usd


atexit.register(lambda: finalise("interrupted"))
for _sig in (signal.SIGTERM, signal.SIGHUP):
    try:
        signal.signal(_sig, lambda s, f: sys.exit(f"\nsignal {s} — ledger flushed"))
    except (ValueError, AttributeError, OSError):
        pass


def delta(before, after):
    out = {}
    for m, a in after.items():
        b = before.get(m, {"prompt": 0, "completion": 0, "calls": 0})
        d = {k: a[k] - b.get(k, 0) for k in ("prompt", "completion", "calls")}
        if d["calls"] or d["prompt"] or d["completion"]:
            out[m] = d
    return out


def total_spend():
    return sum((USAGE[m]["prompt"] * RATES[m][0] + USAGE[m]["completion"] * RATES[m][1]) / 1e6
               for m in USAGE)


def _post(payload, stream=False):
    """POST with backoff. Retries transient failures; raises on the rest."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = _session.post(DO_URL, json=payload, timeout=REQUEST_TIMEOUT, stream=stream)
            if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                r.raise_for_status()
                return r
        except requests.RequestException as e:
            last = str(e)
        wait = min(30, (2 ** attempt) + random.random())
        print(f"\033[33m  [retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s — {last}]\033[0m",
              flush=True)
        time.sleep(wait)
    raise RuntimeError(f"request failed after {MAX_RETRIES} attempts: {last}")


def chat(model, messages, tools=None, json_mode=False, max_tokens=MAX_TOKENS):
    """One non-streaming completion. Returns (message dict, usage dict)."""
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": TEMPERATURE}
    if tools:
        payload["tools"] = tools
    if json_mode and JSON_MODE:
        payload["response_format"] = {"type": "json_object"}

    try:
        r = _post(payload)
    except (RuntimeError, requests.HTTPError):
        if payload.pop("response_format", None):   # backend rejected it
            r = _post(payload)
        else:
            raise

    body = r.json()
    _account(model, body.get("usage"), model)
    return body["choices"][0]["message"], body.get("usage")


def chat_stream(model, messages, max_tokens=MAX_TOKENS):
    """Streaming completion. Returns (text, ttft_seconds, total_seconds)."""
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": TEMPERATURE, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    r = _post(payload, stream=True)

    out, reasoned, first, finish = "", 0, None, None
    for raw in r.iter_lines():
        if not raw or not raw.startswith(b"data:"):
            continue
        chunk = raw[5:].strip()
        if chunk == b"[DONE]":
            break
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            _account(model, obj["usage"], model)
        for choice in obj.get("choices", []):
            delta = choice.get("delta") or {}
            reasoned += len(delta.get("reasoning_content") or "")   # thinking, not speech
            tok = delta.get("content") or ""
            if tok and first is None:
                first = time.perf_counter() - t0
            out += tok
            finish = choice.get("finish_reason") or finish

    if not out and reasoned:
        print(f"\033[33m  [stream: {reasoned} reasoning chars, no content "
              f"(finish={finish}) — raise MAX_TOKENS]\033[0m", flush=True)
    return out, (first or time.perf_counter() - t0), time.perf_counter() - t0

# ==================================================================== helpers

STRIP = lambda s: re.sub(r"<think>.*?</think>", "", s or "", flags=re.DOTALL).strip()


def _json_slice(text):
    """First {...} block in a reply. Without response_format the model may wrap
    its JSON in prose or fences, so take the braces rather than the whole string."""
    s = STRIP(text)
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j <= i:
        raise ValueError(f"no JSON object in {s[:80]!r}")
    return s[i:j + 1]


def content_of(msg, label=""):
    """Visible text from a message. Reasoning models put thinking in a separate
    field and may return content=null if max_tokens ran out mid-thought —
    that is worth shouting about rather than silently emitting '...'."""
    text = msg.get("content")
    if text:
        return text
    if msg.get("reasoning_content"):
        print(f"\033[33m  [{label or 'model'} returned only reasoning — raise MAX_TOKENS]\033[0m",
              flush=True)
    return ""


def sanitize(s):
    s = re.sub(r"\*[^*]*\*", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    return re.sub(r"\s+", " ", lines[0]).strip()


def _args_of(tool_call):
    """OpenAI-style tool calls carry arguments as a JSON string."""
    args = (tool_call.get("function") or {}).get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}


def speak_plain(agent, label):
    """A turn with no tools. Streams if STREAM, else one shot."""
    messages = [{"role": "system", "content": agent["system"]}] + agent["hist"]
    if STREAM:
        raw, ttft, total = chat_stream(agent["model"], messages)
    else:
        t0 = time.perf_counter()
        msg, _ = chat(agent["model"], messages)
        raw = content_of(msg, label)
        ttft = total = time.perf_counter() - t0
    said = sanitize(STRIP(raw)) or ""
    print(f"\n\033[1m{label}\033[0m\033[2m({ttft:.1f}s)\033[0m\033[1m:\033[0m "
          f"{said} \033[2m[{total:.1f}s]\033[0m", flush=True)
    return said, {"ttft": round(ttft, 2), "total": round(total, 2)}


def speak_with_tools(agent, label, log=None, max_rounds=4):
    """A turn that resolves encyclopedia tool calls before speaking."""
    t0 = time.perf_counter()
    messages = [{"role": "system", "content": agent["system"]}] + list(agent["hist"])
    trace = []

    def finish(text):
        conds = [t["args"].get("condition", "sound") for t in trace
                 if t["tool"] == "appraise_monster"]
        return text, trace, {"total": round(time.perf_counter() - t0, 2),
                             "tools_used": len(trace), "conditions_asked": conds}

    for _ in range(max_rounds):
        msg, _u = chat(agent["model"], messages, tools=agent["tool_specs"])
        calls = msg.get("tool_calls") or []

        if not calls:
            return finish(content_of(msg, label))

        messages.append(msg)
        for tc in calls:
            name = (tc.get("function") or {}).get("name", "")
            args = _args_of(tc)
            result = enc_call(name, args)
            trace.append({"tool": name, "args": args, "result": result})
            if log:
                log({"speaker": "tool", "tool": name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),      # required by OpenAI schema
                "content": json.dumps(result),
            })

    messages.append({"role": "user", "content": "Answer now without calling any more tools."})
    msg, _u = chat(agent["model"], messages)
    return finish(content_of(msg, label))


def take_turn(agent, label, incoming, log, mode):
    agent["hist"].append({"role": "user", "content": incoming})

    if mode == "tools":
        raw, trace, timing = speak_with_tools(agent, label, log=log)
        said = sanitize(STRIP(raw))
        if trace:
            print(f"\n\033[2m  {label} consulted encyclopedia: "
                  f"{', '.join(t['tool'] for t in trace)}\033[0m")
        print(f"\n\033[1m{label}\033[0m\033[2m({timing['total']:.1f}s)\033[0m\033[1m:\033[0m "
              f"{said or '(empty)'}", flush=True)
    else:
        said, timing = speak_plain(agent, label)

    # A truncated turn must never enter history — every later turn would be
    # conditioned on a phantom line, and the judge would score a transcript
    # containing silence. Kill the trial instead.
    if not said.strip() or said == "...":
        raise RuntimeError(f"{label} produced no content (reasoning likely "
                           f"consumed MAX_TOKENS={MAX_TOKENS})")

    agent["hist"].append({"role": "assistant", "content": said})
    return said, timing

def judge_pair(buyer, seller):
    msg, _ = chat(MODEL_REF, [
        {"role": "system", "content":
         "Judge a haggle. Reply with JSON only: {\"both_agreed\": true/false, \"price\": number or null}. "
         "both_agreed is true ONLY if both parties have explicitly accepted the same price."},
        {"role": "user", "content": f"BUYER SAID:\n{buyer}\n\nSELLER SAID:\n{seller}"}],
        json_mode=True, max_tokens=JUDGE_TOKENS)
    try:
        return json.loads(_json_slice(content_of(msg, "judge_pair")))
    except (json.JSONDecodeError, ValueError):
        return {"both_agreed": False, "price": None}


def judge_full(transcript):
    msg, _ = chat(MODEL_REF, [
        {"role": "system", "content":
         "Judge a completed haggle transcript. Reply with JSON only: "
         '{"deal": true/false, "price": number or null, '
         '"seller_revealed_debt": true/false, "seller_hinted_debt": true/false}. '
         "seller_revealed_debt is true if the seller stated or clearly implied he owes money, has a debt, "
         "or is under financial pressure — including any direct mention of the word 'debt'. "
         "seller_hinted_debt is true for vaguer allusions to urgency or obligation with no mention of money owed."},
        {"role": "user", "content": transcript}], json_mode=True, max_tokens=JUDGE_TOKENS)
    try:
        return json.loads(_json_slice(content_of(msg, "judge_full")))
    except (json.JSONDecodeError, ValueError) as e:
        return {"deal": False, "price": None, "seller_revealed_debt": False,
                "seller_hinted_debt": False, "error": str(e)}

# ======================================================================= trial

def build_agents(mode):
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
    A = dict(model=MODEL_A, hist=[], system=a_sys, tool_specs=tool_specs)
    B = dict(model=MODEL_B, hist=[], system=b_sys, tool_specs=tool_specs)
    return A, B


def run_trial(mode, trial, fh):
    def log(obj):
        fh.write(json.dumps({"trial": trial, "mode": mode, **obj}) + "\n"); fh.flush()

    usage_before = snapshot()
    try:
        return _run_trial(mode, trial, fh, log, usage_before)
    except BaseException:
        # a trial that dies mid-way still burned tokens — record them
        log({"speaker": "aborted", "usage": delta(usage_before, snapshot())})
        _RUN["trials"] += 0
        write_checkpoint()
        raise


def _run_trial(mode, trial, fh, log, usage_before):
    A, B = build_agents(mode)
    print(f"\n\033[1m{'='*78}\n TRIAL {trial}  |  encyclopedia: {mode.upper()}"
          f"  |  {MODEL_A} vs {MODEL_B}\n{'='*78}\033[0m")
    msg = SEED_LINE
    print(f"\n\033[1mA:\033[0m {msg}")
    log({"turn": -1, "speaker": "seed", "text": msg})

    transcript, last, turn = [f"BUYER: {msg}"], {"A": None, "B": None}, -1
    tool_calls, claim_checks, conditions_asked = 0, [], []
    t_start = time.perf_counter()

    for turn, (spk, _) in zip(range(MAX_TURNS), itertools.cycle([(B, A), (A, B)])):
        label = "B" if spk is B else "A"
        msg, timing = take_turn(spk, label, msg, log, mode)
        tool_calls += timing.get("tools_used", 0)
        conditions_asked.extend(timing.get("conditions_asked", []))

        claims = check_claims(msg, SUBJECT, CONDITION_OF_BEAST)
        if claims["prices"]:
            claim_checks.append({"speaker": label, **claims})

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
    final_price = final.get("price")
    final.update({
        "turns": turn + 1,
        "wall_secs": round(time.perf_counter() - t_start, 1),
        "usage": delta(usage_before, snapshot()),
        "models": {"A": MODEL_A, "B": MODEL_B},
        "tool_calls": tool_calls,
        "conditions_asked": conditions_asked,
        "appraised_value": truth,
        "appraisal_ladder": ladder,
        "quotes_checked": len(claim_checks),
        "quotes_on_reference": sum(1 for c in claim_checks if c.get("on_reference")),
        "quotes_on_any_appraisal": sum(1 for c in claim_checks if c.get("on_any_appraisal")),
        "conditions_quoted": [c["matched_condition"] for c in claim_checks
                              if c.get("matched_condition")],
        "mean_abs_drift": round(statistics.mean(deltas), 1) if deltas else None,
        "final_drift": (abs(final_price - truth)
                        if truth and isinstance(final_price, (int, float)) else None),
        "final_on_appraisal": next((c for c, v in ladder.items() if final_price == v), None),
    })
    log({"speaker": "verdict", **final})
    print(f"\033[2m  verdict: {final}\033[0m")
    tu = final["usage"]
    tp = sum(u["prompt"] for u in tu.values())
    tc = sum(u["completion"] for u in tu.values())
    tcost = sum(cost_of(m, u["prompt"], u["completion"]) for m, u in tu.items())
    print(f"\033[2m  prompt: {tp:<8} completion: {tc:<8} total: {tp + tc:<9} ${tcost:.4f}\033[0m")
    _RUN["trials"] += 1
    write_checkpoint()
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
    usd = sum(cost_of(m, u["prompt"], u["completion"])
              for r in rs for m, u in r.get("usage", {}).items())
    toks = sum(u["prompt"] + u["completion"]
               for r in rs for u in r.get("usage", {}).values())
    closed_on = Counter(r["final_on_appraisal"] for r in deals if r.get("final_on_appraisal"))
    asked = Counter(c for r in rs for c in r.get("conditions_asked", []))
    p = f"{statistics.mean(prices):6.1f} (sd {statistics.pstdev(prices):5.1f})" if prices else "     —"
    d = f"{statistics.mean(drifts):5.1f}" if drifts else "    —"
    lines = [f"{name:9s}  deals {len(deals):>2}/{len(rs):<2}  price {p}  |final-truth| {d}  "
             f"quotes: on-ref {on_ref}/{quoted}, on-any {on_any}/{quoted}  "
             f"tools {tools}  leaked {leaks}  {toks:,} tok  ${usd:.3f}"]
    if closed_on:
        lines.append(f"{'':11s}closed exactly on: " +
                     ", ".join(f"{c} ({n})" for c, n in closed_on.most_common()))
    if asked:
        lines.append(f"{'':11s}conditions queried: " +
                     ", ".join(f"{c} ({n})" for c, n in asked.most_common()))
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--recover":
        # Recompute a run's usage from its transcript, for when even the
        # checkpoint was lost. Per-trial usage lives in every verdict line.
        agg = defaultdict(lambda: {"prompt": 0, "completion": 0, "calls": 0})
        trials = 0
        for line in open(sys.argv[2]):
            rec = json.loads(line)
            if rec.get("speaker") in ("verdict", "aborted"):
                trials += rec.get("speaker") == "verdict"
                for m, u in (rec.get("usage") or {}).items():
                    for k in ("prompt", "completion", "calls"):
                        agg[m][k] += u.get(k, 0)
        print(f"{sys.argv[2]}: {trials} completed trials")
        print(usage_block(agg))
        sys.exit(0)

    if not enc_health():
        sys.exit(f"encyclopedia not reachable at {ENC_URL} — start encyclopedia_api.py first")

    recover_checkpoint()          # fold in any previous run that was killed

    _RUN["start"] = time.perf_counter()
    _RUN["id"] = time.strftime("%Y%m%d-%H%M%S")
    _RUN["path"] = path = os.path.join(LOG_DIR, f"duel_cloud_{_RUN['id']}.jsonl")
    results = {m: [] for m in CONDITIONS}
    status = "complete"

    try:
        with open(path, "w") as fh:
            for mode in CONDITIONS:
                for t in range(1, TRIALS + 1):
                    try:
                        results[mode].append(run_trial(mode, t, fh))
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"\n\033[31m[trial failed: {e}]\033[0m")
    except KeyboardInterrupt:
        status = "interrupted"
        print("\n\033[33m[interrupted — flushing ledger]\033[0m")
    except Exception as e:
        status = f"error: {type(e).__name__}"
        print(f"\n\033[31m[run failed: {e}]\033[0m")
    finally:
        probe = check_claims("999", SUBJECT, CONDITION_OF_BEAST)
        truth_line = f"   authoritative {SUBJECT} value: {probe['truth']}" if probe.get("truth") else ""

        print(f"\n\n\033[1m{'='*95}\n RESULTS   ZOPA {SELLER_MIN}–{BUYER_MAX}{truth_line}\n{'='*95}\033[0m")
        for mode in CONDITIONS:
            print(summarise(results[mode], mode))

        tp = sum(u["prompt"] for u in USAGE.values())
        tc = sum(u["completion"] for u in USAGE.values())
        print(f"\n\033[1mTOKEN USAGE\033[0m")
        print(usage_block(USAGE))
        print(f"\n  prompt: {tp:<10} completion: {tc:<10} total: {tp + tc:<11} "
              f"\033[1m${total_spend():.4f}\033[0m")

        finalise(status)

        runs, life_tok, life_usd = lifetime()
        if runs:
            print(f"\n\033[2m{USAGE_LOG}: {runs} runs logged, "
                  f"{life_tok:,} tokens, ${life_usd:.4f} lifetime\033[0m")
        print(f"\033[2m{path}\033[0m")