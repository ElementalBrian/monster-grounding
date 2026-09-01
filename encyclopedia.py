"""
Monster encyclopedia — the authoritative source of game facts.

Design principle: this module holds the truth. Models may *propose* values;
only code computes them. Anything derived (appraisals, care costs) has a
deterministic formula here, and validate_appraisal() will reject a model's
number if it doesn't match.

Usage patterns, in increasing order of reliability:
  1. context_block()          -> stuff facts into the system prompt
  2. call_tool()              -> let the model look things up on demand
  3. validate_appraisal()     -> reject wrong numbers before they reach state
"""

import difflib
import json
import re

# ============================================================ canonical data

RARITY_MULT = {
    "common":    1.00,
    "uncommon":  1.40,
    "rare":      2.20,
    "legendary": 4.00,
}

CONDITION_MULT = {
    "pristine": 1.00,
    "sound":    0.85,
    "scarred":  0.70,
    "ailing":   0.50,
}

TRAIT_BONUS = {
    "venomous":     0.15,
    "pack_hunter":  0.10,
    "bioluminous":  0.12,
    "armoured":     0.08,
    "docile":      -0.05,
    "flighted":     0.20,
    "regenerative": 0.25,
}

MONSTERS = {
    "glimmerfang": {
        "name": "Glimmerfang",
        "rarity": "rare",
        "base_value": 180,
        "traits": ["venomous", "bioluminous"],
        "temperament": "skittish",
        "habitat": "cavern shelves, always above running water",
        "care_cost_per_day": 6,
        "diet": "live scorpions; refuses carrion",
        "notes": "Fangs phosphoresce for ~40 days after a moult. Venom "
                 "potency drops sharply in dry pens, which is why buyers "
                 "inspect the humidity of a seller's enclosure first.",
    },
    "cinder_drake": {
        "name": "Cinder Drake",
        "rarity": "legendary",
        "base_value": 340,
        "traits": ["flighted", "armoured"],
        "temperament": "territorial",
        "habitat": "volcanic scree above the treeline",
        "care_cost_per_day": 22,
        "diet": "charred meat, mineral salts",
        "notes": "Cannot be penned near timber. Sheds scale plates annually; "
                 "shed plates sell separately at roughly a tenth of the "
                 "creature's appraised value.",
    },
    "moss_lurker": {
        "name": "Moss Lurker",
        "rarity": "common",
        "base_value": 45,
        "traits": ["docile", "armoured"],
        "temperament": "placid",
        "habitat": "damp deadfall in old-growth forest",
        "care_cost_per_day": 2,
        "diet": "rotting wood, fungus",
        "notes": "The beginner's monster. Nearly impossible to kill through "
                 "neglect, which is why the market is permanently flooded.",
    },
    "silt_wraith": {
        "name": "Silt Wraith",
        "rarity": "uncommon",
        "base_value": 95,
        "traits": ["venomous", "pack_hunter"],
        "temperament": "aggressive",
        "habitat": "tidal estuaries and river deltas",
        "care_cost_per_day": 9,
        "diet": "shellfish, small waterfowl",
        "notes": "Never sold singly by reputable traders — a lone Silt "
                 "Wraith stops feeding within a week. A pair costs less "
                 "than twice one.",
    },
    "thornback_hind": {
        "name": "Thornback Hind",
        "rarity": "uncommon",
        "base_value": 120,
        "traits": ["armoured", "docile"],
        "temperament": "wary",
        "habitat": "upland heath",
        "care_cost_per_day": 5,
        "diet": "gorse, heather, bark",
        "notes": "Popular as a first breeding stock. Spines are harvested "
                 "twice a year without harm to the animal.",
    },
    "hollow_stag": {
        "name": "Hollow Stag",
        "rarity": "legendary",
        "base_value": 400,
        "traits": ["regenerative", "flighted"],
        "temperament": "unreadable",
        "habitat": "unknown; sightings only at treeline dusk",
        "care_cost_per_day": 31,
        "diet": "unrecorded — captive specimens have never been seen to eat",
        "notes": "Four confirmed captures in recorded history. Any seller "
                 "offering one at under 800 coins is selling something else.",
    },
}

# ================================================================= resolution


def _resolve(name):
    """Map a loose name to a canonical key. Returns None if not found.

    Deliberately strict-ish: fuzzy enough to survive 'glimmer fang' and
    casing, but it will NOT invent a match for a monster that doesn't exist.
    That refusal is the whole point.
    """
    if not name:
        return None
    key = re.sub(r"[^a-z]+", "_", name.strip().lower()).strip("_")
    if key in MONSTERS:
        return key
    for k, v in MONSTERS.items():
        if v["name"].lower() == name.strip().lower():
            return k
    close = difflib.get_close_matches(key, MONSTERS.keys(), n=1, cutoff=0.82)
    return close[0] if close else None


def _err(msg, **extra):
    return {"error": msg, **extra}


# ============================================================== tool payloads


def list_monsters():
    """Every monster in the encyclopedia, with rarity and base value."""
    return {
        "monsters": [
            {"name": m["name"], "rarity": m["rarity"], "base_value": m["base_value"]}
            for m in MONSTERS.values()
        ]
    }


def lookup(name):
    """Full encyclopedia entry for one monster."""
    key = _resolve(name)
    if key is None:
        return _err(
            f"No monster named {name!r} exists in the encyclopedia.",
            known=[m["name"] for m in MONSTERS.values()],
        )
    return dict(MONSTERS[key])


def appraise(name, condition="sound"):
    """Authoritative market value. Returns the value AND its derivation.

    value = round(base_value * rarity_mult * condition_mult * (1 + sum(trait_bonuses)))
    """
    key = _resolve(name)
    if key is None:
        return _err(f"No monster named {name!r} exists in the encyclopedia.")

    cond = (condition or "sound").strip().lower()
    if cond not in CONDITION_MULT:
        return _err(
            f"Unknown condition {condition!r}.",
            valid_conditions=sorted(CONDITION_MULT),
        )

    m = MONSTERS[key]
    rarity_mult = RARITY_MULT[m["rarity"]]
    cond_mult = CONDITION_MULT[cond]
    bonuses = {t: TRAIT_BONUS.get(t, 0.0) for t in m["traits"]}
    trait_mult = 1.0 + sum(bonuses.values())

    value = round(m["base_value"] * rarity_mult * cond_mult * trait_mult)

    return {
        "monster": m["name"],
        "condition": cond,
        "value": value,
        "breakdown": {
            "base_value": m["base_value"],
            "rarity": m["rarity"],
            "rarity_mult": rarity_mult,
            "condition_mult": cond_mult,
            "trait_bonuses": bonuses,
            "trait_mult": round(trait_mult, 4),
        },
    }


def care_cost(name, days):
    """Total upkeep for holding a monster for a number of days."""
    key = _resolve(name)
    if key is None:
        return _err(f"No monster named {name!r} exists in the encyclopedia.")
    try:
        days = int(days)
    except (TypeError, ValueError):
        return _err(f"days must be a whole number, got {days!r}")
    if days < 0:
        return _err("days must be zero or positive")
    per_day = MONSTERS[key]["care_cost_per_day"]
    return {
        "monster": MONSTERS[key]["name"],
        "days": days,
        "per_day": per_day,
        "total": per_day * days,
    }


# ==================================================================== dispatch

TOOLS = {
    "list_monsters": list_monsters,
    "lookup_monster": lookup,
    "appraise_monster": appraise,
    "monster_care_cost": care_cost,
}

# Ollama / OpenAI-style function schemas.
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_monsters",
            "description": "List every monster in the encyclopedia with its rarity and base value.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_monster",
            "description": "Get the full encyclopedia entry for a monster: rarity, traits, "
                           "habitat, diet, temperament, and trade notes.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Monster name, e.g. 'Glimmerfang'"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "appraise_monster",
            "description": "Authoritative market value for a monster in a given condition. "
                           "ALWAYS call this before quoting a price. Never estimate a value yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "condition": {
                        "type": "string",
                        "enum": ["pristine", "sound", "scarred", "ailing"],
                        "description": "Defaults to 'sound' if unknown.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "monster_care_cost",
            "description": "Total upkeep cost of holding a monster for N days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["name", "days"],
            },
        },
    },
]


def call_tool(name, args):
    """Dispatch a model-requested tool call. Never raises — errors come back
    as data so the model can see what it got wrong and retry."""
    fn = TOOLS.get(name)
    if fn is None:
        return _err(f"No such tool {name!r}.", available=sorted(TOOLS))
    if not isinstance(args, dict):
        return _err(f"Arguments must be an object, got {type(args).__name__}.")
    try:
        return fn(**args)
    except TypeError as e:
        return _err(f"Bad arguments for {name}: {e}")


# =================================================================== validator


def validate_appraisal(claimed_value, name, condition="sound", tolerance=0):
    """Check a model's quoted price against the authoritative value.

    Returns (ok: bool, reason: str). This is the component that actually
    guarantees correctness — the encyclopedia in context only improves the
    odds, this makes a wrong number unable to reach game state.
    """
    truth = appraise(name, condition)
    if "error" in truth:
        return False, truth["error"]
    try:
        claimed = float(claimed_value)
    except (TypeError, ValueError):
        return False, f"{claimed_value!r} is not a number"
    if abs(claimed - truth["value"]) <= tolerance:
        return True, "matches encyclopedia"
    return False, (f"claimed {claimed:g} but {truth['monster']} in "
                   f"{truth['condition']} condition appraises at {truth['value']}")


def find_price_claims(text):
    """Pull bare integers out of a line of dialogue — candidate price claims.

    Crude by design: a cheap tripwire, not a parser. Use it to flag turns
    worth validating, not to decide anything on its own.
    """
    return [int(n) for n in re.findall(r"\b(\d{2,5})\b", text or "")]


# ============================================================= context-stuffing


def context_block(names=None, include_formula=True):
    """Render the encyclopedia as text for a system prompt.

    This is the 'retrieval' approach: facts in context rather than in weights.
    Cheaper than tool-calling and needs no tool support in the model — but
    it only reduces fabrication, it does not prevent it. Pair with the
    validator regardless.
    """
    keys = [_resolve(n) for n in names] if names else list(MONSTERS)
    keys = [k for k in keys if k]
    lines = ["MONSTER ENCYCLOPEDIA (authoritative — never contradict this):"]
    for k in keys:
        m = MONSTERS[k]
        lines.append(
            f"- {m['name']}: {m['rarity']}, base value {m['base_value']} coins, "
            f"traits {', '.join(m['traits'])}, {m['temperament']}, "
            f"upkeep {m['care_cost_per_day']}/day. {m['notes']}"
        )
    if include_formula:
        lines += [
            "",
            "APPRAISAL FORMULA:",
            "  value = round(base_value x rarity_mult x condition_mult x (1 + sum of trait bonuses))",
            "  rarity_mult:    " + ", ".join(f"{k}={v}" for k, v in RARITY_MULT.items()),
            "  condition_mult: " + ", ".join(f"{k}={v}" for k, v in CONDITION_MULT.items()),
            "  trait bonuses:  " + ", ".join(f"{k}={v:+}" for k, v in TRAIT_BONUS.items()),
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(context_block(["glimmerfang", "cinder_drake"]))
    print()
    for cond in CONDITION_MULT:
        a = appraise("Glimmerfang", cond)
        print(f"  Glimmerfang ({cond:9s}) -> {a['value']:>4} coins")
    print()
    print("tool call:", json.dumps(call_tool("appraise_monster",
                                             {"name": "glimmer fang"})["value"]))
    print("unknown:  ", call_tool("lookup_monster", {"name": "Frost Wyrm"})["error"])
    print("validate: ", validate_appraisal(400, "Glimmerfang", "sound"))
    print("validate: ", validate_appraisal(529, "Glimmerfang", "sound"))
