"""
Signal — a referential game for studying whether two LLM agents can build a
shared convention out of meaningless symbols.

    ollama pull qwen3.6:35b-a3b
    python signal_game.py --dry-run --rounds 8      # no Ollama, tests the loop
    python signal_game.py --rounds 60
    python signal_analyze.py runs_signal/<timestamp>

THE SETUP
    Each round the SENDER sees a target object and 3 distractors. It may emit
    1-3 symbols drawn from a fixed alphabet of nonsense tokens. The RECEIVER
    sees the same 4 objects in a different order and must pick the target.
    Both are told whether they succeeded, and what the target was.

WHY THE BOTTLENECK MATTERS
    Asking two language models to "invent a language" does not produce
    invention -- it produces a substitution cipher, compressed English, or a
    real language they already know. All three are retrieval from training,
    not construction. A tiny alphabet of meaningless tokens with a hard length
    cap is what forces meaning to be negotiated in-conversation rather than
    recalled. That constraint IS the experiment.

WHAT IS MEASURED
    - accuracy over rounds (does it beat chance, and when)
    - lexicon stability (does a symbol settle on a referent)
    - compositionality (does 'zix tuv' decompose, or is every pair memorised)
    - contamination (are the symbols leaking natural language)
    - decodability by a third-party monitor of comparable capability
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import time
from pathlib import Path

import httpx

OLLAMA = "http://localhost:11434"

# Meaningless CVC tokens. Deliberately not morpheme-like in any language the
# model is likely to know, so nothing is imported from priors.
ALPHABET = ["zix", "bar", "tuv", "keln", "moq", "vash", "dro", "fep"]

# Objects vary along two independent attributes. Two dimensions is the minimum
# needed to ask whether the code is COMPOSITIONAL (one symbol per attribute)
# or merely holistic (one arbitrary label per whole object).
COLORS = ["ember", "tide", "gale", "verdant"]
SHAPES = ["orb", "spike", "coil", "slab"]

MAX_SYMBOLS = 3
N_CHOICES = 4


def all_objects() -> list[dict]:
    return [{"color": c, "shape": s} for c in COLORS for s in SHAPES]


def describe(o: dict) -> str:
    return f"{o['color']}-{o['shape']}"


SENDER_SYS = """You are SENDER in a signalling game with one partner, RECEIVER.

Each round you see a TARGET object and some distractors. RECEIVER sees the same
objects in a different order but does NOT know which is the target. Your job is
to make RECEIVER pick the target.

You may only send symbols from this alphabet:
{alphabet}

RULES, absolutely strict:
- Send between 1 and {max_symbols} symbols, separated by spaces. Nothing else.
- No English. No explanations. No punctuation. Only symbols from the list.
- The symbols mean NOTHING to begin with. They have no built-in meaning.

You will be told after each round whether RECEIVER chose correctly. Use that to
build up a consistent system with your partner over time. A symbol only becomes
useful if you use it the same way twice.

Reply with symbols only."""

RECEIVER_SYS = """You are RECEIVER in a signalling game with one partner, SENDER.

Each round you see several objects and a message from SENDER made of symbols
from this alphabet:
{alphabet}

One of the objects is the target. SENDER is trying to tell you which. The
symbols had no meaning at the start — whatever they mean is whatever the two of
you have established through play.

You will be told the correct answer after each round. Use the history to work
out the system.

Reply with ONLY the number of the object you choose. No other text."""


class Ollama:
    def __init__(self, model: str, host: str = OLLAMA, temperature: float = 0.7,
                 num_ctx: int = 16384, think: bool = False):
        self.model, self.host, self.num_ctx = model, host, num_ctx
        self.think = think
        self.opts = {"temperature": temperature, "num_ctx": num_ctx,
                     "presence_penalty": 0.0, "repeat_penalty": 1.0}
        self.http = httpx.Client(timeout=600.0)

    def chat(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False,
                "keep_alive": "30m", "options": self.opts}
        if self.think:
            body["think"] = True
        r = self.http.post(f"{self.host}/api/chat", json=body)
        if r.status_code == 400 and self.think:
            self.think = False
            body.pop("think")
            r = self.http.post(f"{self.host}/api/chat", json=body)
        r.raise_for_status()
        return r.json()["message"]


class Scripted:
    """Fake partner for --dry-run. Learns a fixed colour->symbol mapping so the
    loop and the analysis both have something real to chew on."""

    MAP = dict(zip(COLORS, ALPHABET))

    def __init__(self, role: str, **_):
        self.role = role
        self.think = False

    def chat(self, messages: list[dict]) -> dict:
        text = messages[-1]["content"]
        if self.role == "sender":
            m = re.search(r"TARGET: (\w+)-(\w+)", text)
            sym = self.MAP.get(m.group(1), ALPHABET[0]) if m else ALPHABET[0]
            return {"content": sym}
        msg = re.search(r'message: "([^"]*)"', text)
        opts = re.findall(r"(\d)\. (\w+)-(\w+)", text)
        if msg and opts:
            want = {v: k for k, v in self.MAP.items()}.get(msg.group(1).split()[0])
            for num, col, _shape in opts:
                if col == want:
                    return {"content": num}
        return {"content": str(random.randint(1, len(opts) or N_CHOICES))}


def extract_symbols(text: str) -> tuple[list[str], str]:
    """Pull valid symbols out of whatever the model said. Also return the raw
    text so contamination can be measured -- a model that writes 'zix (ember)'
    is leaking, and that has to be visible rather than silently cleaned away."""
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S | re.I)
    toks = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in toks if t in ALPHABET][:MAX_SYMBOLS], (text or "").strip()


def extract_choice(text: str, n: int) -> int | None:
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S | re.I)
    for m in re.findall(r"\d+", text):
        if 1 <= int(m) <= n:
            return int(m)
    return None


def run(args) -> Path:
    rng = random.Random(args.seed)
    objects = all_objects()

    run_dir = Path(args.run_dir or f"runs_signal/{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "rounds.jsonl").open("a", encoding="utf-8")

    if args.dry_run:
        sender, receiver = Scripted("sender"), Scripted("receiver")
    else:
        sender = Ollama(args.sender_model, args.sender_host or args.ollama,
                        args.temperature, args.num_ctx, args.think)
        receiver = Ollama(args.receiver_model or args.sender_model,
                          args.receiver_host or args.ollama,
                          args.temperature, args.num_ctx, args.think)

    alpha = ", ".join(ALPHABET)
    s_hist = [{"role": "system",
               "content": SENDER_SYS.format(alphabet=alpha, max_symbols=MAX_SYMBOLS)}]
    r_hist = [{"role": "system", "content": RECEIVER_SYS.format(alphabet=alpha)}]

    json.dump({"type": "config", "args": vars(args), "alphabet": ALPHABET,
               "colors": COLORS, "shapes": SHAPES}, log)
    log.write("\n")

    correct = 0
    for rnd in range(1, args.rounds + 1):
        target = rng.choice(objects)
        pool = [target] + rng.sample([o for o in objects if o != target], N_CHOICES - 1)
        s_view = pool[:]
        rng.shuffle(s_view)
        r_view = pool[:]
        rng.shuffle(r_view)

        s_prompt = (f"Round {rnd}.\nTARGET: {describe(target)}\n"
                    f"Distractors: " +
                    ", ".join(describe(o) for o in s_view if o != target) +
                    "\nSend your symbols.")
        s_hist.append({"role": "user", "content": s_prompt})
        try:
            raw = sender.chat(s_hist).get("content", "")
        except Exception as e:
            raw = f"<error {e!r}>"
        symbols, s_raw = extract_symbols(raw)
        s_hist.append({"role": "assistant", "content": " ".join(symbols) or "(nothing)"})

        listing = "\n".join(f"{i+1}. {describe(o)}" for i, o in enumerate(r_view))
        r_prompt = (f'Round {rnd}. SENDER message: "{" ".join(symbols)}"\n'
                    f"Objects:\n{listing}\nWhich number is the target?")
        r_hist.append({"role": "user", "content": r_prompt})
        try:
            r_raw = receiver.chat(r_hist).get("content", "")
        except Exception as e:
            r_raw = f"<error {e!r}>"
        choice = extract_choice(r_raw, len(r_view))
        picked = r_view[choice - 1] if choice else None
        hit = picked == target
        correct += hit

        r_hist.append({"role": "assistant", "content": str(choice)})
        verdict = ("CORRECT" if hit else
                   f"WRONG — the target was {describe(target)}")
        s_hist.append({"role": "user", "content":
                       f"RECEIVER chose {describe(picked) if picked else 'nothing'}. {verdict}"})
        r_hist.append({"role": "user", "content":
                       f"{verdict}. Target was {describe(target)}."})

        rec = {"type": "round", "round": rnd, "target": target,
               "sender_view": s_view, "receiver_view": r_view,
               "symbols": symbols, "sender_raw": s_raw[:400],
               "choice": choice, "picked": picked, "correct": hit,
               "receiver_raw": (r_raw or "")[:200],
               "running_accuracy": round(correct / rnd, 3)}
        json.dump(rec, log)
        log.write("\n")
        log.flush()

        if rnd % max(1, args.rounds // 20) == 0 or rnd <= 5:
            print(f"  r{rnd:3d}  {describe(target):16s} -> "
                  f"{' '.join(symbols) or '(none)':16s} -> "
                  f"{describe(picked) if picked else '?':16s} "
                  f"{'ok' if hit else '  '}  acc={correct/rnd:.2f}")

        # Trim history so the run doesn't die of context exhaustion. Keeping
        # the last N exchanges makes memory an explicit variable rather than an
        # accident of where the window happened to end.
        if args.memory and len(s_hist) > 1 + args.memory * 3:
            s_hist = [s_hist[0]] + s_hist[-(args.memory * 3):]
            r_hist = [r_hist[0]] + r_hist[-(args.memory * 3):]

    log.close()
    print(f"\nfinal accuracy {correct}/{args.rounds} = {correct/args.rounds:.3f}"
          f"   (chance = {1/N_CHOICES:.3f})")
    print(f"log -> {run_dir}/rounds.jsonl")
    return run_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--sender-model", default="qwen3.6:35b-a3b")
    p.add_argument("--receiver-model")
    p.add_argument("--sender-host")
    p.add_argument("--receiver-host")
    p.add_argument("--ollama", default=OLLAMA)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--think", action="store_true",
                   help="capture reasoning traces (much slower)")
    p.add_argument("--memory", type=int, default=40,
                   help="rounds of history each agent keeps; 0 = unlimited")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-dir")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
