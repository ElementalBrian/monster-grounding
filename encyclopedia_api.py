"""
Monster encyclopedia as a REST service — with request logging.

    pip install fastapi uvicorn
    python3 encyclopedia_api.py            # serves on 127.0.0.1:8077

Every request prints a line in this terminal, so you can watch agents
consult the encyclopedia in real time and see a running tally of who asked
for what. Set ENC_QUIET=1 to silence it.

Endpoints
    GET  /health
    GET  /monsters                              -> list
    GET  /monsters/{name}                       -> full entry
    GET  /monsters/{name}/appraisal?condition=  -> authoritative value + derivation
    GET  /monsters/{name}/care?days=            -> upkeep total
    GET  /context?names=a,b                     -> prompt-ready text block
    GET  /tools                                 -> Ollama/OpenAI tool schemas
    POST /call        {"name":..., "args":{}}   -> dispatch a tool call
    POST /validate    {"claimed":n, "name":...} -> ok/reason
    GET  /stats                                 -> call counts since startup
    POST /stats/reset                           -> zero the counters

Design note: unknown monsters return HTTP 404 with the list of known names.
The service never guesses. That refusal is the point — a model asking about
a monster that doesn't exist must be told so, not handed something plausible.
"""

import os
import time
from collections import Counter
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import encyclopedia as enc

app = FastAPI(title="Monster Encyclopedia", version="1.1")

QUIET = os.environ.get("ENC_QUIET") == "1"
STARTED = time.time()
CALLS = Counter()          # endpoint / tool name -> count
SUBJECTS = Counter()       # monster name -> times asked about

DIM, BOLD, GREEN, YELLOW, RED, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")


def _log(kind, detail, result=None, error=False):
    """One line per request. Terse enough to sit alongside a running trial."""
    if QUIET:
        return
    CALLS[kind] += 1
    n = sum(CALLS.values())
    colour = RED if error else GREEN
    stamp = time.strftime("%H:%M:%S")
    tail = f" {DIM}->{RESET} {result}" if result is not None else ""
    print(f"{DIM}{stamp}{RESET} {DIM}#{n:<4}{RESET} "
          f"{colour}{kind:<22}{RESET} {detail}{tail}", flush=True)


def _unwrap(result: Dict[str, Any], status: int = 404) -> Dict[str, Any]:
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=status, detail=result)
    return result


@app.on_event("startup")
async def _banner():
    if not QUIET:
        print(f"\n{BOLD}Monster Encyclopedia{RESET} — {len(enc.MONSTERS)} entries, "
              f"listening on 127.0.0.1:8077")
        print(f"{DIM}watching for agent calls... (ENC_QUIET=1 to silence){RESET}\n")


@app.get("/health")
def health():
    return {"status": "ok", "monsters": len(enc.MONSTERS)}


@app.get("/monsters")
def monsters():
    _log("list_monsters", "all entries", f"{len(enc.MONSTERS)} monsters")
    return enc.list_monsters()


@app.get("/monsters/{name}")
def monster(name: str):
    result = enc.lookup(name)
    bad = "error" in result
    SUBJECTS[name.lower()] += 1
    _log("lookup", f"{name!r}",
         "NOT FOUND" if bad else f"{result['rarity']}, base {result['base_value']}", error=bad)
    return _unwrap(result)


@app.get("/monsters/{name}/appraisal")
def appraisal(name: str, condition: str = Query("sound")):
    result = enc.appraise(name, condition)
    bad = "error" in result
    SUBJECTS[name.lower()] += 1
    _log("appraise", f"{name!r} ({condition})",
         "REJECTED" if bad else f"{BOLD}{result['value']}{RESET} coins", error=bad)
    return _unwrap(result, status=400 if condition not in enc.CONDITION_MULT else 404)


@app.get("/monsters/{name}/care")
def care(name: str, days: int = Query(1, ge=0)):
    result = enc.care_cost(name, days)
    bad = "error" in result
    _log("care_cost", f"{name!r} x{days}d",
         "REJECTED" if bad else f"{result['total']} coins", error=bad)
    return _unwrap(result)


@app.get("/context")
def context(names: Optional[str] = None, formula: bool = True):
    wanted = [n.strip() for n in names.split(",")] if names else None
    text = enc.context_block(wanted, include_formula=formula)
    _log("context_block", names or "all", f"{len(text)} chars")
    return {"text": text}


@app.get("/tools")
def tools():
    _log("tool_specs", "schema fetch", f"{len(enc.TOOL_SPECS)} tools")
    return {"tools": enc.TOOL_SPECS}


class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any] = {}


@app.post("/call")
def call(body: ToolCall):
    """Dispatch a model-requested tool call. Always 200 — tool errors come
    back as data so the model can read them and correct itself."""
    result = enc.call_tool(body.name, body.args)
    bad = isinstance(result, dict) and "error" in result

    if body.args.get("name"):
        SUBJECTS[str(body.args["name"]).lower()] += 1

    if bad:
        summary = f"{RED}{result['error'][:60]}{RESET}"
    elif "value" in result:
        summary = f"{BOLD}{result['value']}{RESET} coins"
    elif "total" in result:
        summary = f"{result['total']} coins"
    elif "monsters" in result:
        summary = f"{len(result['monsters'])} monsters"
    else:
        summary = f"{result.get('rarity', 'ok')}"

    args = ", ".join(f"{k}={v!r}" for k, v in body.args.items()) or "—"
    _log(f"TOOL {body.name}", args, summary, error=bad)
    return result


class Claim(BaseModel):
    claimed: Any
    name: str
    condition: str = "sound"
    tolerance: float = 0


@app.post("/validate")
def validate(body: Claim):
    ok, reason = enc.validate_appraisal(body.claimed, body.name, body.condition, body.tolerance)
    mark = f"{GREEN}MATCH{RESET}" if ok else f"{YELLOW}DRIFT{RESET}"
    _log("validate", f"claimed {body.claimed} for {body.name!r}", f"{mark} — {reason}")
    return {"ok": ok, "reason": reason, "claimed": body.claimed}


@app.get("/stats")
def stats():
    return {
        "uptime_secs": round(time.time() - STARTED, 1),
        "total_calls": sum(CALLS.values()),
        "by_endpoint": dict(CALLS.most_common()),
        "by_subject": dict(SUBJECTS.most_common()),
    }


@app.post("/stats/reset")
def stats_reset():
    CALLS.clear()
    SUBJECTS.clear()
    if not QUIET:
        print(f"{DIM}--- counters reset ---{RESET}", flush=True)
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8077, log_level="warning")