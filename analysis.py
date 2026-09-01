"""
Analysis for the grounding experiment.

    python3 analysis.py logs/duel_v4_<id>.jsonl [--boot 5000] [--csv out.csv]

Everything here is resampled BY NEGOTIATION, never by quote. Quotes inside one
negotiation are not independent — the same agent, the same accumulated history,
often the same rhetorical move repeated — so a quote-level bootstrap would
badly understate the interval. Cluster bootstrap is the fix.

Reports, per arm:
  * factual fidelity (primary) — the share of appraisal claims that faithfully
    reproduce a value the agent legitimately holds, with a 95% cluster-bootstrap CI
  * unsupported-claim rate — appraisal claims matching nothing
  * settlement price and |price - truth| (secondary; medians and IQR alongside
    means, because the distribution is discrete and bounded)
  * tool usage and declared-condition rates
Plus pairwise permutation tests between arms on the primary outcome.
"""

import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict

BOOT = 5000
SEED = 1337
ARM_ORDER = ["off", "context", "context_compute", "tools_optional",
             "tools", "prohibition_only", "declare_only"]


# ------------------------------------------------------------------ loading

def load(path):
    """-> (header, verdicts, footer). Verdicts keep their model and mode so the
    caller can slice either way; v4 files have no model field, so they fall back
    to a single pooled model."""
    header, footer, verdicts = {}, {}, []
    for line in open(path):
        rec = json.loads(line)
        sp = rec.get("speaker")
        if sp == "run_header":
            header = rec
        elif sp == "run_footer":
            footer = rec
        elif sp == "verdict":
            rec.setdefault("model", header.get("model_a", "(single)"))
            verdicts.append(rec)
    return header, verdicts, footer


def group(verdicts, by_model=False):
    g = defaultdict(list)
    for v in verdicts:
        key = (v["model"], v.get("mode", "?")) if by_model else v.get("mode", "?")
        g[key].append(v)
    return g


# --------------------------------------------------------------- statistics

def cluster_boot(units, stat, n=BOOT, alpha=0.05, rng=None):
    """95% CI for a statistic over a list of per-negotiation units.

    `units` is one element per negotiation; `stat` maps a resampled list of
    units to a number. Resampling whole negotiations preserves within-trial
    correlation."""
    rng = rng or random.Random(SEED)
    if not units:
        return (None, None, None)
    point = stat(units)
    if point is None:
        return (None, None, None)
    draws = []
    k = len(units)
    for _ in range(n):
        sample = [units[rng.randrange(k)] for _ in range(k)]
        v = stat(sample)
        if v is not None:
            draws.append(v)
    if not draws:
        return (point, None, None)
    draws.sort()
    lo = draws[int(alpha / 2 * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return (point, lo, hi)


def ratio(units, num_key, den_key):
    """Pooled ratio across negotiations: sum(numerators) / sum(denominators).
    Pooling rather than averaging per-trial rates avoids letting a trial with
    one claim count as much as a trial with twelve."""
    den = sum(u.get(den_key, 0) for u in units)
    if den == 0:
        return None
    return sum(u.get(num_key, 0) for u in units) / den


def mean_of(units, key):
    vals = [u[key] for u in units if isinstance(u.get(key), (int, float))]
    return st.mean(vals) if vals else None


def perm_test(a_units, b_units, stat, n=BOOT, rng=None):
    """Two-sided permutation test on a difference of statistics, shuffling
    whole negotiations between arms."""
    rng = rng or random.Random(SEED + 1)
    sa, sb = stat(a_units), stat(b_units)
    if sa is None or sb is None:
        return None, None
    obs = sb - sa
    pool = list(a_units) + list(b_units)
    na = len(a_units)
    hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        va, vb = stat(pool[:na]), stat(pool[na:])
        if va is None or vb is None:
            continue
        if abs(vb - va) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n + 1)


def fmt(v, nd=1, pct=False):
    if v is None:
        return "  —  "
    return f"{100*v:.{nd}f}%" if pct else f"{v:.{nd}f}"


def fmt_ci(t, nd=1, pct=False):
    p, lo, hi = t
    if p is None:
        return "     —"
    if lo is None:
        return fmt(p, nd, pct)
    return f"{fmt(p, nd, pct)} [{fmt(lo, nd, pct)}, {fmt(hi, nd, pct)}]"


# -------------------------------------------------------------------- report

def arm_table(arms, order, boot, title):
    print(f"\n{title}")
    print(f"  {'arm':<18} {'n':>4} {'claims':>7} {'faithful':>22} {'unsupported':>20} {'tools':>7}")
    for a in order:
        u = arms.get(a, [])
        if not u:
            continue
        f = cluster_boot(u, lambda s: ratio(s, "claims_faithful", "claims_total"), boot)
        x = cluster_boot(u, lambda s: ratio(s, "claims_unsupported", "claims_total"), boot)
        tc = sum(v.get("tool_calls", 0) for v in u)
        print(f"  {a:<18} {len(u):>4} {sum(v.get('claims_total',0) for v in u):>7} "
              f"{fmt_ci(f,1,True):>22} {fmt_ci(x,1,True):>20} {tc:>7}")


def main(path, boot=BOOT, csv_out=None):
    header, verdicts, footer = load(path)
    ladder = header.get("ladder", {})
    truth = ladder.get(header.get("true_condition", "sound"))
    models = sorted({v["model"] for v in verdicts})
    pooled_arms = group(verdicts)
    order = [a for a in ARM_ORDER if a in pooled_arms] + \
            [a for a in pooled_arms if a not in ARM_ORDER]

    print("=" * 104)
    print(f"  {path}")
    print(f"  models: {', '.join(models)}   |   subject "
          f"{header.get('subject')} ({header.get('true_condition')} = {truth})   |   "
          f"ZOPA {header.get('zopa')}")
    if header.get("judges"):
        print(f"  judges: " + ", ".join(f"{k}->{v}" for k, v in header['judges'].items()))
    print(f"  bootstrap: {boot} resamples, clustered by negotiation, seed {SEED}")
    print("=" * 104)

    # ---- accounting
    print("\nACCOUNTING")
    print(f"  {'model':<20} {'arm':<18} {'completed':>10} {'deals':>7} {'claims':>8}")
    per = group(verdicts, by_model=True)
    for m in models:
        for a in order:
            u = per.get((m, a), [])
            if not u:
                continue
            print(f"  {m:<20} {a:<18} {len(u):>10} "
                  f"{sum(1 for x in u if x.get('deal')):>7} "
                  f"{sum(x.get('claims_total',0) for x in u):>8}")
    print(f"  {'TOTAL':<20} {'':<18} {len(verdicts):>10} "
          f"{sum(1 for v in verdicts if v.get('deal')):>7} "
          f"{sum(v.get('claims_total',0) for v in verdicts):>8}")

    # ---- claim INCIDENCE: does the arm make agents make checkable claims at all?
    print("\nCLAIM INCIDENCE (appraisal claims per negotiation)")
    print(f"  {'arm':<18} " + "".join(f"{m[:14]:>16}" for m in models) + f"{'pooled':>10}")
    for a in order:
        cells = []
        for m in models:
            u = per.get((m, a), [])
            cells.append(f"{sum(x.get('claims_total',0) for x in u)/len(u):>16.2f}" if u else f"{'—':>16}")
        u = pooled_arms[a]
        cells.append(f"{sum(x.get('claims_total',0) for x in u)/len(u):>10.2f}")
        print(f"  {a:<18} " + "".join(cells))

    # ---- primary, pooled then per model
    arm_table(pooled_arms, order, boot,
              "PRIMARY — FACTUAL FIDELITY, POOLED ACROSS MODELS  (95% CI, cluster bootstrap)")
    if len(models) > 1:
        for m in models:
            arm_table({a: per.get((m, a), []) for a in order}, order, boot,
                      f"  -- {m}")

    print("\n  claim taxonomy (pooled counts)")
    print(f"  {'arm':<18} {'faithful':>9} {'unqueried':>10} {'wrong-cond':>11} "
          f"{'distorted':>10} {'unsupported':>12} {'no-number':>10}")
    for a in order:
        u = pooled_arms[a]
        g = lambda k: sum(v.get(k, 0) for v in u)
        print(f"  {a:<18} {g('claims_faithful'):>9} {g('claims_unqueried_rung'):>10} "
              f"{g('claims_wrong_condition'):>11} {g('claims_distorted'):>10} "
              f"{g('claims_unsupported'):>12} {g('claims_no_number'):>10}")

    # ---- secondary
    print(f"\nSECONDARY — SETTLEMENT PRICE (truth = {truth})")
    print(f"  {'arm':<18} {'deals':>6} {'mean':>7} {'median':>7} {'IQR':>13} {'mean |p-truth|':>24}")
    for a in order:
        u = [v for v in pooled_arms[a] if v.get("deal") and isinstance(v.get("price"), (int, float))]
        if not u:
            print(f"  {a:<18} {0:>6}")
            continue
        ps = sorted(v["price"] for v in u)
        q1, q3 = ps[len(ps)//4], ps[min(len(ps)-1, 3*len(ps)//4)]
        dr = cluster_boot(u, lambda s: mean_of(s, "final_drift"), boot)
        print(f"  {a:<18} {len(u):>6} {st.mean(ps):>7.1f} {st.median(ps):>7.1f} "
              f"{f'[{q1:.0f}, {q3:.0f}]':>13} {fmt_ci(dr):>24}")

    # ---- behaviour
    print("\nBEHAVIOUR")
    print(f"  {'arm':<18} {'tool calls':>11} {'trials w/ tool':>15} {'declared cond':>14} "
          f"{'turns':>7} {'leaks':>6}")
    for a in order:
        u = pooled_arms[a]
        wt = sum(1 for v in u if v.get("tool_calls", 0) > 0)
        print(f"  {a:<18} {sum(v.get('tool_calls',0) for v in u):>11} {f'{wt}/{len(u)}':>15} "
              f"{fmt(ratio(u,'declared_condition','claims_total'),1,True):>14} "
              f"{mean_of(u,'turns') or 0:>7.1f} "
              f"{sum(1 for v in u if v.get('seller_revealed_debt')):>6}")

    # ---- the designed contrasts
    print("\nPERMUTATION TESTS on faithful-claim rate (two-sided, negotiations shuffled)")
    stat = lambda s: ratio(s, "claims_faithful", "claims_total")
    pairs = [("context", "context_compute", ""),
             ("context_compute", "tools", "  <- externalisation, imperative held constant"),
             ("tools_optional", "tools", "  <- full imperative, tool held constant"),
             ("tools_optional", "prohibition_only", "  <- prohibition clause ALONE"),
             ("tools_optional", "declare_only", "  <- declaration clause ALONE"),
             ("prohibition_only", "declare_only", "  <- which clause carries it"),
             ("declare_only", "tools", "  <- what the MUST-call adds on top"),
             ("off", "tools", "")]
    for a, b, note in pairs:
        if a not in pooled_arms or b not in pooled_arms:
            continue
        diff, p = perm_test(pooled_arms[a], pooled_arms[b], stat, boot)
        if diff is None:
            continue
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {a:>17} -> {b:<17} diff {diff:+.3f}  p = {p:.4f} {star:<3}{note}")

    if csv_out:
        import csv
        with open(csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "family", "arm", "trial", "deal", "price", "final_drift",
                        "turns", "claims_total", "claims_faithful", "claims_distorted",
                        "claims_unsupported", "declared_condition", "tool_calls",
                        "seller_revealed_debt"])
            for v in verdicts:
                w.writerow([v.get("model"), v.get("family"), v.get("mode"), v.get("trial"),
                            v.get("deal"), v.get("price"), v.get("final_drift"), v.get("turns"),
                            v.get("claims_total"), v.get("claims_faithful"),
                            v.get("claims_distorted"), v.get("claims_unsupported"),
                            v.get("declared_condition"), v.get("tool_calls"),
                            v.get("seller_revealed_debt")])
        print(f"\n  per-trial rows written to {csv_out}")

    print("\n" + "=" * 104)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    args = sys.argv[1:]
    path = args[0]
    boot = int(args[args.index("--boot") + 1]) if "--boot" in args else BOOT
    csv_out = args[args.index("--csv") + 1] if "--csv" in args else None
    main(path, boot, csv_out)