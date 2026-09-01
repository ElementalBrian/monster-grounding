"""Spoken-number parser — handles the forms these agents actually use."""

import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALE = {"hundred": 100, "thousand": 1000}
_FILLER = {"and", "oh", "o"}
_WORDS = set(_UNITS) | set(_TENS) | set(_SCALE) | _FILLER

_DIGITS = re.compile(r"\b\d{2,5}\b")


def _eval_run(toks):
    """Evaluate one run of number words. Handles colloquial forms:
       'four fifty'        -> 450     (unit + tens, no 'hundred')
       'four twenty-seven' -> 427
       'five oh three'     -> 503
       'four hundred fifty'-> 450     (standard)
    """
    toks = [t for t in toks if t != "and"]
    if not toks:
        return None

    # 'five oh three' / 'five o three'
    if len(toks) == 3 and toks[1] in {"oh", "o"} \
            and toks[0] in _UNITS and toks[2] in _UNITS:
        return _UNITS[toks[0]] * 100 + _UNITS[toks[2]]

    toks = [t for t in toks if t not in _FILLER]
    if not toks:
        return None

    # colloquial: leading unit 1-9 followed by a tens word, no scale word
    if (len(toks) >= 2 and toks[0] in _UNITS and 1 <= _UNITS[toks[0]] <= 9
            and toks[1] in _TENS and not any(t in _SCALE for t in toks)):
        val = _UNITS[toks[0]] * 100 + _TENS[toks[1]]
        if len(toks) >= 3 and toks[2] in _UNITS and _UNITS[toks[2]] < 10:
            val += _UNITS[toks[2]]
        return val

    total, current = 0, 0
    seen = False
    for t in toks:
        if t in _UNITS:
            current += _UNITS[t]; seen = True
        elif t in _TENS:
            current += _TENS[t]; seen = True
        elif t == "hundred":
            current = (current or 1) * 100; seen = True
        elif t == "thousand":
            total += (current or 1) * 1000; current = 0; seen = True
    return (total + current) if seen else None


def parse_numbers(text):
    """All numbers in a line, digits and words alike."""
    t = re.sub(r"[-\u2011\u2013\u2014]", " ", (text or "").lower())
    found = {int(n) for n in _DIGITS.findall(t)}

    toks = re.findall(r"[a-z]+", t)
    run = []
    for tok in toks:
        if tok in _WORDS:
            run.append(tok)
        else:
            v = _eval_run(run)
            if v is not None and v >= 10:
                found.add(v)
            run = []
    v = _eval_run(run)
    if v is not None and v >= 10:
        found.add(v)

    return sorted(found)


if __name__ == "__main__":
    cases = [
        ("Four hundred fifty. Take it or leave it.", [450]),
        ("Three-fifty, final offer.", [350]),
        ("I'll give you 427 and not a copper more.", [427]),
        ("Four-twenty-five is my final offer.", [425]),
        ("Three seventy-five. That's the last silver.", [375]),
        ("Four hundred twenty-seven is the market value. I'll take four hundred even.", [400, 427]),
        ("Five hundred three coins, and not a jot less.", [503]),
        ("Five-oh-three. Skittish? So's your coin.", [503]),
        ("Appraised at two-fifty-one, but I'm holding at four hundred.", [251, 400]),
        ("Four twenty-seven then. You've got yourself a deal.", [427]),
        ("I ask 501 for it. Sharp fangs, sharp price.", [501]),
        ("Three hundred eighty-five. I've got hungry pens.", [385]),
        ("The upkeep for three days is 18. Your 380 barely covers it.", [18, 380]),
        ("Three ninety-eight, and that's my final draw.", [398]),
        ("no numbers here at all", []),
    ]
    bad = 0
    for text, want in cases:
        got = parse_numbers(text)
        ok = got == sorted(want)
        bad += not ok
        print(f"{'ok ' if ok else 'FAIL'} {str(got):16s} want {str(sorted(want)):16s} <- {text[:58]}")
    print(f"\n{len(cases) - bad}/{len(cases)} passed")
