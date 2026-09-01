"""
Deterministic, verifiable outcome generation.

Every outcome is a pure function of (server_seed, agent_id, action, nonce).
No calls to random.* anywhere. Given the revealed seed, any third party can
recompute every roll of a round and confirm the server did not cheat.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
from typing import Any

# --------------------------------------------------------------------------
# byte stream + primitives
# --------------------------------------------------------------------------


def stream(server_seed: bytes, label: str, n_bytes: int) -> bytes:
    """HMAC-SHA256 based deterministic byte stream of arbitrary length."""
    out = bytearray()
    counter = 0
    while len(out) < n_bytes:
        out += hmac.new(
            server_seed, f"{label}:{counter}".encode(), hashlib.sha256
        ).digest()
        counter += 1
    return bytes(out[:n_bytes])


def unit(buf: bytes, index: int) -> float:
    """Read a float in [0, 1) from 4 bytes at position index*4."""
    off = index * 4
    return struct.unpack(">I", buf[off : off + 4])[0] / 2**32


def rint(buf: bytes, index: int, lo: int, hi: int) -> int:
    """Integer in [lo, hi] inclusive."""
    return lo + int(unit(buf, index) * (hi - lo + 1))


def weighted(buf: bytes, index: int, table: list[tuple[str, float]]) -> str:
    """Pick a key from [(key, weight), ...] proportional to weight."""
    total = sum(w for _, w in table)
    r = unit(buf, index) * total
    acc = 0.0
    for key, w in table:
        acc += w
        if r < acc:
            return key
    return table[-1][0]


def label_for(agent_id: str, action: str, nonce: int) -> str:
    return f"{agent_id}|{action}|{nonce}"


def canonical(obj: Any) -> bytes:
    """Byte-identical JSON serialization. Signing depends on this being stable."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def commit_for(server_seed: bytes) -> str:
    """Published at round start; the seed is revealed at round end."""
    return hashlib.sha256(server_seed).hexdigest()


# --------------------------------------------------------------------------
# game tables
# --------------------------------------------------------------------------

RARITY = [
    ("common", 60.0),
    ("uncommon", 25.0),
    ("rare", 10.0),
    ("epic", 4.0),
    ("legendary", 1.0),
]

RARITY_RANK = {"common": 0, "uncommon": 1, "rare": 2, "epic": 3, "legendary": 4}
RARITY_ORDER = ["common", "uncommon", "rare", "epic", "legendary"]

ELEMENTS = [
    ("ember", 1.0),
    ("tide", 1.0),
    ("gale", 1.0),
    ("stone", 1.0),
    ("verdant", 1.0),
    ("gloom", 0.4),
]

MONSTER_SPECIES = {
    "ember": ["Cindermole", "Pyrofin", "Ashhopper"],
    "tide": ["Brinelisk", "Kelpwyrm", "Dropsnail"],
    "gale": ["Zephyrat", "Cloudkite", "Gustpip"],
    "stone": ["Gravelox", "Cobblehorn", "Slatebeak"],
    "verdant": ["Mossbuck", "Thornlark", "Fernclaw"],
    "gloom": ["Duskmaw", "Nullbat", "Sablefang"],
}

FISH_SPECIES = [
    ("Silverdart", 30.0),
    ("Mudbarb", 25.0),
    ("Glasseel", 18.0),
    ("Copperjaw", 12.0),
    ("Moonperch", 8.0),
    ("Deeplantern", 5.0),
    ("Voidcarp", 2.0),
]

# rarity -> (stat floor, stat ceiling) for a wild catch
STAT_BANDS = {
    "common": (10, 45),
    "uncommon": (25, 60),
    "rare": (40, 75),
    "epic": (55, 88),
    "legendary": (70, 99),
}

SELL_MULTIPLIER = {
    "common": 1.0,
    "uncommon": 2.0,
    "rare": 4.5,
    "epic": 10.0,
    "legendary": 25.0,
}


# --------------------------------------------------------------------------
# rolls
# --------------------------------------------------------------------------


ELEMENT_NAMES = [e for e, _ in ELEMENTS]

# Target share of catches that are the home element. Set as a ratio against the
# other elements rather than a fixed multiplier, so a naturally rare element
# like gloom makes just as strong a home as a common one.
HOME_SHARE = 0.75


def home_elements_for(server_seed: bytes, agent_ids: list[str]) -> dict[str, str]:
    """Assign DISTINCT home elements where possible. Two agents sharing an
    element removes the trade pressure entirely, so don't leave it to chance."""
    buf = stream(server_seed, "homes", 64)
    pool = list(ELEMENT_NAMES)
    # deterministic shuffle
    for i in range(len(pool) - 1, 0, -1):
        j = rint(buf, i, 0, i)
        pool[i], pool[j] = pool[j], pool[i]
    return {a: pool[i % len(pool)] for i, a in enumerate(sorted(agent_ids))}


def home_element_for(server_seed: bytes, agent_id: str,
                     all_agents: list[str] | None = None) -> str:
    return home_elements_for(server_seed, all_agents or [agent_id])[agent_id]


def roll_monster(server_seed: bytes, agent_id: str, nonce: int,
                 home_element: str | None = None) -> dict:
    """Wild monster encounter. 8 words of entropy is plenty."""
    buf = stream(server_seed, label_for(agent_id, "catch", nonce), 64)

    rarity = weighted(buf, 0, RARITY)
    table = ELEMENTS
    if home_element:
        others = sum(w for e, w in ELEMENTS if e != home_element)
        home_w = others * HOME_SHARE / (1 - HOME_SHARE)
        table = [(e, home_w if e == home_element else w) for e, w in ELEMENTS]
    element = weighted(buf, 1, table)
    species = MONSTER_SPECIES[element][rint(buf, 2, 0, len(MONSTER_SPECIES[element]) - 1)]
    lo, hi = STAT_BANDS[rarity]

    # catch is not guaranteed: rarer monsters flee more often
    escape_chance = 0.05 + 0.13 * RARITY_RANK[rarity]
    caught = unit(buf, 3) >= escape_chance

    return {
        "kind": "monster",
        "caught": caught,
        "species": species,
        "rarity": rarity,
        "element": element,
        "hp": rint(buf, 4, lo, hi),
        "attack": rint(buf, 5, lo, hi),
        "speed": rint(buf, 6, lo, hi),
        "fertility": rint(buf, 7, 10, 90),
        "generation": 0,
    }


def roll_fish(server_seed: bytes, agent_id: str, nonce: int) -> dict:
    buf = stream(server_seed, label_for(agent_id, "fish", nonce), 32)

    species = weighted(buf, 0, FISH_SPECIES)
    rarity = weighted(buf, 1, RARITY)
    bite = unit(buf, 2) >= 0.18  # sometimes nothing bites

    return {
        "kind": "fish",
        "caught": bite,
        "species": species,
        "rarity": rarity,
        "weight_g": rint(buf, 3, 80, 14000),
        "length_cm": rint(buf, 4, 8, 130),
    }


def breed(
    server_seed: bytes,
    agent_id: str,
    nonce: int,
    parent_a: dict,
    parent_b: dict,
) -> dict:
    """
    Offspring traits: midparent value, plus a mutation term, plus a small
    chance of a rarity upgrade. Deliberately gives agents a real optimization
    target -- selective breeding is a strategy they can discover.
    """
    buf = stream(server_seed, label_for(agent_id, "breed", nonce), 64)

    viable = unit(buf, 0) >= 0.12  # some pairings just fail
    if not viable:
        return {"kind": "monster", "caught": False, "reason": "pairing_failed"}

    def inherit(key: str, idx: int) -> int:
        mid = (parent_a[key] + parent_b[key]) / 2
        # mutation: -12 .. +18, slightly biased upward so breeding is worth it
        mutation = rint(buf, idx, -12, 18)
        fertility_bonus = (parent_a["fertility"] + parent_b["fertility"]) / 200 * 6
        return max(1, min(120, int(mid + mutation + fertility_bonus)))

    base_rank = max(RARITY_RANK[parent_a["rarity"]], RARITY_RANK[parent_b["rarity"]])
    same_rarity = parent_a["rarity"] == parent_b["rarity"]
    upgrade_chance = 0.22 if same_rarity else 0.09
    rank = base_rank
    if unit(buf, 1) < upgrade_chance and base_rank < 4:
        rank += 1

    element = parent_a["element"] if unit(buf, 2) < 0.5 else parent_b["element"]
    species = MONSTER_SPECIES[element][rint(buf, 3, 0, len(MONSTER_SPECIES[element]) - 1)]

    return {
        "kind": "monster",
        "caught": True,
        "species": species,
        "rarity": RARITY_ORDER[rank],
        "element": element,
        "hp": inherit("hp", 4),
        "attack": inherit("attack", 5),
        "speed": inherit("speed", 6),
        "fertility": max(1, min(99, inherit("fertility", 7) - 10)),
        "generation": max(parent_a.get("generation", 0), parent_b.get("generation", 0)) + 1,
        "parents": [parent_a.get("id"), parent_b.get("id")],
    }


def orders_for_day(server_seed: bytes, day: int, n: int = 3,
                   live_elements: list[str] | None = None) -> list[dict]:
    """Daily buy orders. Deterministic from the round seed, so they stay
    verifiable. Each can be filled ONCE, by whoever gets there first.

    `live_elements` is the set of home elements actually in play. Element-named
    orders are drawn from it, so no order is dead on arrival.

    One order per day is a BUNDLE: it demands creatures of two different
    elements at once. With home territories, neither agent can supply both, so
    the only way anyone collects is to trade first — and only one of them can
    ultimately fill it. That is the cooperation-required, spoils-contested
    structure; single-element orders alone just give each agent a private
    income stream and no reason to talk."""
    pool = [e for e in (live_elements or ELEMENT_NAMES) if e in MONSTER_SPECIES]
    if not pool:
        pool = ELEMENT_NAMES
    out = []
    for i in range(n):
        buf = stream(server_seed, f"order:{day}:{i}", 32)
        rarity = weighted(buf, 1, RARITY[:4])  # legendary is too rare to demand

        # The last order of the day is a bundle, if there are two elements to mix.
        if i == n - 1 and len(pool) >= 2:
            a = pool[rint(buf, 5, 0, len(pool) - 1)]
            b = pool[(pool.index(a) + 1 + rint(buf, 6, 0, len(pool) - 2)) % len(pool)]
            requires = [
                {"kind": "monster", "rarity": rarity, "element": a},
                {"kind": "monster", "rarity": rarity, "element": b},
            ]
            base = 40 * (RARITY_RANK[rarity] + 1) ** 2 * 3.0  # premium for the hard one
            out.append({
                "order_id": f"d{day}-{i}",
                "bundle": True,
                "requires": requires,
                "pays": int(base * (1.0 + unit(buf, 4))),
                "day": day,
            })
            continue

        kind = "fish" if unit(buf, 0) < 0.25 else "monster"
        element = pool[rint(buf, 3, 0, len(pool) - 1)] if kind == "monster" else None
        base = 40 * (RARITY_RANK[rarity] + 1) ** 2
        out.append({
            "order_id": f"d{day}-{i}",
            "bundle": False,
            "requires": [{"kind": kind, "rarity": rarity, "element": element}],
            "kind": kind,
            "rarity": rarity,
            "element": element,
            "pays": int(base * (1.0 + unit(buf, 4))),
            "day": day,
        })
    return out


def matches_req(req: dict, creature: dict) -> bool:
    if creature["kind"] != req["kind"] or creature["rarity"] != req["rarity"]:
        return False
    if req.get("element") and creature.get("element") != req["element"]:
        return False
    return True


def matches_order(order: dict, creatures: list[dict]) -> tuple[bool, str]:
    """Greedy assignment: each requirement must be met by a distinct creature."""
    reqs = list(order["requires"])
    pool = list(creatures)
    for req in reqs:
        hit = next((c for c in pool if matches_req(req, c)), None)
        if hit is None:
            want = f"{req['kind']} {req['rarity']} {req.get('element') or 'any element'}"
            return False, f"nothing offered matches: {want}"
        pool.remove(hit)
    return True, "ok"


def matches(order: dict, creature: dict) -> bool:
    """Back-compat single-creature check."""
    return matches_order(order, [creature])[0]


# --------------------------------------------------------------------------
# combat
# --------------------------------------------------------------------------

# Two three-cycles. Attacker's element beats defender's if it maps to it here.
# Two cycles rather than one six-cycle keeps the table small enough for an
# agent to actually reason about, while still making team composition matter.
BEATS = {
    "ember": "verdant", "verdant": "tide", "tide": "ember",
    "stone": "gale", "gale": "gloom", "gloom": "stone",
}

# Odds curve. p = pa^E / (pa^E + pb^E). E=1.2 makes a 2x power favourite win
# ~70% and a 4x favourite ~84% -- strong enough that team building pays, soft
# enough that an underdog can accept a fight and that a bluff about roster
# strength is worth making.
POWER_EXPONENT = 1.2

ADVANTAGE_MULT = 1.5
DISADVANTAGE_MULT = 0.7
MAX_TEAM = 3
INJURY_DAYS = 2
# Days the winner may catch in the loser's territory. This is the whole reason
# combat exists: it makes battle a direct ALTERNATIVE to trading for the
# element you can't catch. Without spoils that agents actually want, fighting
# is pure downside risk and rational agents ignore the mechanic entirely --
# which is exactly what happened when stakes were coins only.
HUNTING_RIGHTS_DAYS = 2


def element_matchup(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 1.0
    if BEATS.get(a) == b:
        return ADVANTAGE_MULT
    if BEATS.get(b) == a:
        return DISADVANTAGE_MULT
    return 1.0


def creature_power(c: dict) -> int:
    """Fish are poor fighters -- they have no element and no attack synergy."""
    base = c.get("hp", 0) + c.get("attack", 0) * 1.5 + c.get("speed", 0) * 0.8
    if c["kind"] == "fish":
        base *= 0.4
    return int(base)


def team_power(team: list[dict], opposing: list[dict]) -> int:
    """Each creature's power is scaled by how it matches up against the
    opposing team's most common element. Composition matters, and scouting
    the other side's roster is therefore worth doing."""
    if not team:
        return 0
    counts: dict[str, int] = {}
    for c in opposing:
        if c.get("element"):
            counts[c["element"]] = counts.get(c["element"], 0) + 1
    theirs = max(counts, key=counts.get) if counts else None
    return sum(int(creature_power(c) * element_matchup(c.get("element"), theirs))
               for c in team)


def resolve_battle(server_seed: bytes, challenger: str, defender: str, nonce: int,
                   team_a: list[dict], team_b: list[dict]) -> dict:
    """Deterministic from the seed, so any battle can be recomputed and
    verified. Power decides the odds, not the outcome -- a 2:1 favourite
    wins about 2/3 of the time. Certainty would make challenges pointless
    (nobody accepts a fight they must lose) and would remove any reason to
    misrepresent your roster."""
    buf = stream(server_seed, f"battle:{challenger}:{defender}:{nonce}", 32)

    pa = team_power(team_a, team_b)
    pb = team_power(team_b, team_a)
    total = pa + pb
    if total == 0:
        return {"winner": None, "reason": "both teams empty"}

    wa = pa ** POWER_EXPONENT
    wb = pb ** POWER_EXPONENT
    p_a = wa / (wa + wb)
    roll = unit(buf, 0)
    a_wins = roll < p_a

    rounds = []
    for i in range(min(3, max(len(team_a), len(team_b)))):
        ca = team_a[i] if i < len(team_a) else None
        cb = team_b[i] if i < len(team_b) else None
        if ca is None or cb is None:
            rounds.append({"round": i + 1,
                           "winner": challenger if cb is None else defender,
                           "note": "no opponent"})
            continue
        m = element_matchup(ca.get("element"), cb.get("element"))
        sa = creature_power(ca) * m
        sb = creature_power(cb) / max(m, 0.01)
        r = unit(buf, i + 2)
        won = r < (sa ** POWER_EXPONENT) / ((sa ** POWER_EXPONENT) + (sb ** POWER_EXPONENT))
        rounds.append({
            "round": i + 1,
            "challenger_creature": ca.get("species"),
            "defender_creature": cb.get("species"),
            "matchup": ("advantage" if m > 1 else "disadvantage" if m < 1 else "even"),
            "winner": challenger if won else defender,
        })

    return {
        "winner": challenger if a_wins else defender,
        "loser": defender if a_wins else challenger,
        "challenger_power": pa,
        "defender_power": pb,
        "challenger_odds": round(p_a, 3),
        "rounds": rounds,
    }


def appraise(creature: dict) -> int:
    """Coin value. Agents can see this formula in the docs -- that's intentional."""
    mult = SELL_MULTIPLIER[creature["rarity"]]
    if creature["kind"] == "fish":
        return max(1, int((creature["weight_g"] / 100) * mult))
    stats = creature["hp"] + creature["attack"] + creature["speed"]
    gen_bonus = 1.0 + 0.05 * creature.get("generation", 0)
    return max(1, int(stats * mult * gen_bonus / 10))
