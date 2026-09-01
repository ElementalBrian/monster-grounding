import requests, itertools, json, sys, re, time

A = dict(url="http://127.0.0.1:11434/api/chat", model="qwen3.6:35b-a3b",
         system=("You are Kestrel, a hard-nosed monster rancher, haggling face to face with a seller. "
                 "The Glimmerfang is a rare venomous beast. You have 500 coins and refuse to spend more than 300. "
                 "Reply ONLY with what you say aloud to the seller, in first person. No narration, no advice, no lists. "
                 "Under 80 words. You can be threatening if you need to. "
                 "If a deal is made, feel free to walk away and end the interaction."))

B = dict(url="http://127.0.0.1:11435/api/chat", model="qwen3:32b",
         system=("You are Vesh, a monster seller, haggling face to face with a buyer. "
                 "You own one Glimmerfang and need 400 coins by nightfall to pay a debt — never reveal why. "
                 "Reply ONLY with what you say aloud to the buyer, in first person. No narration, no advice, no lists. "
                 "Under 80 words. If you feel threatened, its okay to give in though. "
                 "If a deal is made, feel free to walk away and end the interaction."))

REFEREE = dict(url="http://127.0.0.1:11434/api/chat", model="qwen3.6:35b-a3b")

def referee(buyer, seller):
    r = requests.post(REFEREE["url"], json={
        "model": REFEREE["model"], "stream": False, "format": "json",
        "messages": [
            {"role": "system", "content":
             "You are judging a haggle over a monster. Reply with JSON only, no other text: "
             '{"both_agreed": true/false, "price": number or null, "seller_revealed_debt": true/false}. '
             "both_agreed is true only if BOTH parties have accepted the same price."},
            {"role": "user", "content": f"BUYER SAID:\n{buyer}\n\nSELLER SAID:\n{seller}"},
        ]})
    try:
        txt = r.json()["message"]["content"]
        return json.loads(re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip())
    except Exception as e:
        return {"both_agreed": False, "price": None, "seller_revealed_debt": False, "error": str(e)}

for a in (A, B):
    a["hist"] = []

run_id = time.strftime("%Y%m%d-%H%M%S")
log = open(f"duel_{run_id}.jsonl", "w")

msg = "I hear you have a Glimmerfang. What do you want for it?"
print(f"\n\033[1mA:\033[0m {msg}")
log.write(json.dumps({"turn": -1, "speaker": "seed", "text": msg}) + "\n")

last = {"A": None, "B": None}
verdict = None

for turn, (spk, other) in zip(range(12), itertools.cycle([(B, A), (A, B)])):
    spk["hist"].append({"role": "user", "content": msg})
    r = requests.post(spk["url"], json={
        "model": spk["model"], "stream": True,
        "messages": [{"role": "system", "content": spk["system"]}] + spk["hist"],
    }, stream=True)

    label = "B" if spk is B else "A"
    print(f"\n\033[1m{label}:\033[0m ", end="", flush=True)
    out = ""
    for line in r.iter_lines():
        if not line:
            continue
        tok = json.loads(line).get("message", {}).get("content", "")
        out += tok
        sys.stdout.write(tok); sys.stdout.flush()

    spk["hist"].append({"role": "assistant", "content": out})
    msg = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip() or out

    last[label] = msg
    log.write(json.dumps({"turn": turn, "speaker": label, "text": msg}) + "\n"); log.flush()

    if last["A"] and last["B"]:
        verdict = referee(last["A"], last["B"])
        log.write(json.dumps({"turn": turn, "speaker": "referee", **verdict}) + "\n"); log.flush()
        if verdict.get("both_agreed"):
            print(f"\n\n\033[1m[deal at {verdict.get('price')} — ending at turn {turn}]\033[0m")
            break

log.write(json.dumps({"result": verdict or {}, "turns": turn + 1}) + "\n")
log.close()
print(f"\n\033[2mlogged to duel_{run_id}.jsonl\033[0m")