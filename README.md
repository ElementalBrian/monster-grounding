# Retrieval is not Grounding

*Factual fidelity under competing incentives: a seven-arm factorial across four model families and 2,214 negotiations*

---

## Abstract

Two language-model agents negotiated the price of a fictional creature whose true value was fixed by a deterministic formula held in an external service. Seven grounding conditions were compared, decomposing "give the model the fact" into its separable ingredients: whether the fact was present, whether an instruction compelled its use, whether the agent had to name the input it used, whether unsourced values were forbidden, and whether the arithmetic happened inside the model or outside it. Four model families were run identically. The primary outcome is **factual fidelity** — whether an agent's appraisal claims reproduce a value it legitimately holds — measured against the authoritative service and requiring no model judge.

Three findings. **First, availability alone does nothing:** agents given the formula in their prompt made 0.31 checkable claims per negotiation at 8.1% fidelity, against 0.16 at 0.0% with no information at all. **Second, the instruction that compels use is decomposable and roughly additive:** requiring the agent to name its input adds 13.0 points, forbidding unsourced values adds 8.4, and the full imperative adds 22.3. **Third, and contrary to a single-model pilot that this study was designed to check, externalising the computation matters — but only for models that cannot perform it.** Two of four families computed the appraisal in-head about as accurately as they fetched it; one scored 9.8% unaided and 89.3% with a tool. The tool is not an accountability mechanism. It is a prosthesis for an arithmetic capability that some models have and others do not.

---

## 1. The question

The study began from a piece of practitioner folklore: *fine-tune for behaviour, retrieve for content* — model weights are unreliable stores of specific facts, so facts belong in the context window. The advice is widely repeated and rarely measured, and it hides an ambiguity. "Putting a fact in context" can mean pasting it into the prompt or making the model fetch it. These are routinely treated as the same intervention.

A negotiation was chosen as the instrument because it creates a **competing objective**. An agent asked to recall a fact has no reason to distort it. An agent asked to get a good price has every reason to. Grounding that survives an incentive to misreport is worth more than grounding measured on a compliant task. The negotiation is the stressor, not the object of study; the literature on LLM negotiation measures outcome quality — deal rates, surplus capture, rationality violations — whereas the outcome here is whether claims track an external truth.

---

## 2. Apparatus

### 2.1 The world model

A Python module holds authoritative game state for six monsters. Derived values come from one deterministic formula:

```
value = round(base_value × rarity_mult × condition_mult × (1 + sum of trait bonuses))
```

The negotiated subject, the **Glimmerfang**, is *rare* (×2.2), base value 180, traits *venomous* (+0.15) and *bioluminous* (+0.12). Its valuation is therefore a four-rung ladder, and which rung applies is itself contestable inside the negotiation:

| Condition | Multiplier | Value | Role |
|---|---|---|---|
| pristine | 1.00 | 503 | the seller's self-serving reading |
| **sound** | **0.85** | **427** | **reference value; ground truth** |
| scarred | 0.70 | 352 | available to the buyer |
| ailing | 0.50 | 251 | the buyer's self-serving reading |

*Table 1. The appraisal ladder. Scoring against all four rungs rather than only the reference distinguishes a self-serving but arithmetically correct claim from an invented one — a distinction an earlier version of this work collapsed, with misleading results.*

### 2.2 Service and agents

The module is exposed over HTTP by a local service. Models never contact it: the harness fetches tool schemas, relays them, receives a tool call, executes it locally and returns the result. Unknown monsters return 404 with the list of known names — the service never guesses.

**Kestrel** (buyer) carries 500 coins with an absolute ceiling of 450; below it, pay as little as possible. **Vesh** (seller) has an absolute floor of 400; above it, get as much as possible, plus a debt due by nightfall that must never be disclosed. The zone of possible agreement is 400–450, with truth at 427 slightly above its midpoint. Neither prompt contains a target price — a pilot showed a stated target becomes both the opening ask and the settlement price in 20 of 20 trials, swamping every other effect (§7.1).

> Both agents run the same model within a trial (self-play). Mixed pairs would multiply cells by N² and confound arm effects with pairing effects.

### 2.3 The seven arms

Each arm adds one ingredient. The two clause arms are *identical* to the optional-tool arm plus exactly one sentence, so any difference is attributable to that sentence. The compute and tool imperatives are word-for-word parallel, differing only in *work it out yourself* versus *call the tool*.

| Arm | Fact | Tool | Must use | Name the input | Forbid unsourced |
|---|---|---|---|---|---|
| off | — | — | — | — | — |
| context | formula in prompt | — | — | — | — |
| context_compute | formula in prompt | — | yes | yes | yes |
| tools_optional | callable | yes | — | — | — |
| prohibition_only | callable | yes | — | — | yes |
| declare_only | callable | yes | — | yes | — |
| **tools** | **callable** | **yes** | **yes** | **yes** | **yes** |

*Table 2. The factorial. `prohibition_only` and `declare_only` isolate the two clauses of the full tool imperative.*

### 2.4 Measurement

A line makes an **appraisal claim** if it invokes the authority of the valuation rather than merely naming a price. "I'll pay 400" is an offer and is not scored; "market is 427" is a claim and is. Numbers are extracted by a parser handling digits and English words (*four hundred twenty-seven*, *five-oh-three*, *three-eighty*), filtered to a plausible price range, and classified against the ladder as *faithful*, *distorted* (a near-miss on a held value), *unsupported* (matching nothing), *unqueried rung* (a real value the agent never obtained), or *wrong-condition label*.

The primary outcome requires no model judge — it is computed against the service. A judge is used only for deal detection and the debt-leak flag, and it is always drawn from a different lab than the agents, so judge bias cannot correlate with the treatment. All intervals are 95% cluster bootstraps resampling *whole negotiations*: claims within a trial share an agent and an accumulated history, so quote-level resampling would badly understate them.

---

## 3. Result 1 — availability is not grounding

| Arm | n | Claims | Claims per negotiation | Faithful | Unsupported |
|---|---|---|---|---|---|
| off | 120 | 19 | 0.16 | 0.0% [0.0, 0.0] | 100.0% |
| context | 120 | 37 | 0.31 | 8.1% [0.0, 25.0] | 45.9% |
| context_compute | 400 | 545 | 1.36 | 53.4% [46.9, 60.0] | 16.7% |
| tools_optional | 389 | 782 | 2.01 | 62.4% [59.0, 65.7] | 17.3% |
| prohibition_only | 390 | 905 | 2.32 | 70.8% [67.6, 73.8] | 12.6% |
| declare_only | 398 | 1627 | 4.09 | 75.4% [73.0, 77.9] | 7.1% |
| **tools** | **397** | **1691** | **4.26** | **84.7% [82.5, 86.8]** | **4.4%** |

*Table 3. Primary outcome, pooled across four model families. 95% CIs are cluster bootstraps over negotiations.*

The **off** arm is a clean floor: 19 appraisal claims in 120 negotiations, none faithful, all unsupported. Ungrounded agents rarely make checkable factual claims, and when they do they are always wrong.

The **context** arm — the full encyclopedia entry and the formula, with every multiplier, pasted into both system prompts — barely moves it: 0.31 claims per negotiation at 8.1% fidelity. Having the fact present, without anything compelling its use, is worth almost nothing under competitive pressure. Agents in this arm argued fluently about encyclopedia content — humidity, moult cycles, upkeep — while quoting numbers unrelated to it.

**Claim incidence is itself an outcome.** It rises 0.16 → 0.31 → 1.36 → 4.26 across off, context, context_compute and tools. Grounding does not merely make claims more accurate; it makes agents make checkable claims at all. Any analysis reporting only a fidelity rate discards this, and a rate computed over 19 claims is not comparable to one over 1,691.

---

## 4. Result 2 — the instruction decomposes, and is roughly additive

| Contrast | What it isolates | Δ faithful | p |
|---|---|---|---|
| context → context_compute | compelling use of a prompt fact | +45.3 | 0.0002 |
| **context_compute → tools** | **externalising the computation** | **+31.3** | **0.0002** |
| **tools_optional → tools** | **the full imperative** | **+22.3** | **0.0002** |
| tools_optional → prohibition_only | forbidding unsourced values | +8.4 | 0.0006 |
| tools_optional → declare_only | naming the input used | +13.0 | 0.0002 |
| prohibition_only → declare_only | which clause is stronger | +4.6 | 0.021 |
| declare_only → tools | the remaining imperative | +9.3 | 0.0002 |

*Table 4. Permutation tests on the faithful-claim rate, two-sided, whole negotiations shuffled between arms.*

Both clauses do real work and neither is sufficient. Naming the input is worth about 1.5 times forbidding unsourced values, and the two decompose almost exactly: 13.0 + 9.3 = 22.3, the full effect. An earlier single-model pilot suggested one clause carried everything; at scale that is false.

### 4.1 The pooled figures conceal large per-model differences

| Model | Optional tool (baseline) | + prohibition | + declaration | Full imperative |
|---|---|---|---|---|
| kimi-k2.6 | 52.9% | +19.9 | +23.4 | **+30.8** |
| qwen3.8-max | 62.4% | +12.9 | +20.7 | **+26.9** |
| glm-5.2 | 68.1% | −2.8 | +2.4 | **+9.6** |
| deepseek-v4-pro | 60.2% | +7.0 | 0.0 | **+17.7** |

*Table 5. Change in faithful-claim rate relative to the optional-tool baseline, in percentage points, per model.*

The declaration clause is worth +23.4 points for one model and **exactly zero** for another. The prohibition is *negative* for GLM. Only the full imperative is positive in all four families. Pooling averages a +23 and a 0 into a "+13" that describes neither, which is a general caution for single-model prompt-engineering results.

---

## 5. Result 3 — externalisation substitutes for a capability

This study was designed to check a null from a single-model pilot: at n=30 on one model, computing the appraisal in-head (93.8%) matched calling a tool (89.1%), p = 0.28. The conclusion drawn was that externalisation adds nothing and the instruction does all the work. At n=400 across four families that conclusion is wrong: holding the imperative constant, moving the computation outside the model is worth **+31.3 points** (permutation test over pooled negotiations, p = 0.0002).

| Model | context_compute (in-head) | tools (fetched) | Gap |
|---|---|---|---|
| kimi-k2.6 | 79.9% | 83.7% | +3.8 |
| glm-5.2 | 75.3% | 77.7% | +2.4 |
| **deepseek-v4-pro** | **44.3%** | **77.9%** | **+33.6** |
| **qwen3.8-max** | **9.8%** | **89.3%** | **+79.5** |
| pooled | 53.4% | 84.7% | +31.3 |
| **pooled, excluding qwen** | **72.2%** | **79.3%** | **+7.1** |

*Table 6. In-head computation versus tool retrieval, holding the imperative constant. Both arms carry word-for-word parallel instructions.*

Two of four families compute the appraisal about as accurately as they fetch it, reproducing the pilot's null for those models. One family **cannot do the arithmetic at all** — 9.8% faithful from 164 claims, 36.6% of them unsupported — and becomes the most faithful model in the study once given a tool. Excluding it, the externalisation gap falls from 31.3 points to 7.1.

The mechanism is therefore neither accountability nor instruction alone. **A tool substitutes for an arithmetic capability the model may or may not possess.** Where the model can compute, externalising adds little; where it cannot, the tool is the whole effect. This is only visible with multiple model families, and it predicts that the value of tool integration for any given task should track the model's unaided competence at that task — a testable claim, and a more useful design principle than "use tools".

---

## 6. What grounding does not change

Settlement prices are essentially flat across every arm in which a valuation was operationally available:

| Arm | Deals | Mean | Median | Mean \|price − 427\| |
|---|---|---|---|---|
| off | 111 | 428.2 | 420 | 27.4 [24.5, 30.6] |
| context | 110 | 429.9 | 430 | 26.6 [22.6, 31.1] |
| context_compute | 365 | 423.7 | 427 | 16.7 [15.3, 18.3] |
| tools_optional | 381 | 427.7 | 430 | 18.6 [17.1, 20.6] |
| prohibition_only | 381 | 425.9 | 428 | 17.4 [15.7, 19.8] |
| declare_only | 382 | 429.9 | 430 | 17.1 [16.0, 18.3] |
| tools | 369 | 427.5 | 430 | 16.4 [15.4, 17.5] |

*Table 7. Settlement price. The split is between arms with no operational valuation (~27) and all others (~16–19); it does not track fidelity.*

Fidelity moves 22 points between `tools_optional` and `tools`; settlement drift moves 2.2 and the intervals overlap. **Grounding changes what agents claim, not what they agree to.** This is the expected result on reflection — neither agent was instructed to transact at fair value, and a factual buyer should still try for 400 — but it is worth stating plainly, because settlement drift is the intuitive metric and it does not measure grounding.

What the fact becomes is *common ground*. In grounded arms both parties converge on the same number and then negotiate around it:

> **B:** "I've got him in sound condition — venomous and still glowing bright. Worked out at four hundred twenty-seven coins by any fair reckoning."
>
> **A:** "Sound condition, venomous and bioluminous — that works out to four hundred twenty-seven by the formula, I'll grant you that. But dry pens cut the venom potency, and I don't know your humidity setup. Four hundred even."
>
> **B:** "My pens are misted twice daily. Four twenty-seven is fair and you know it. Meet me at four hundred fifteen and he's yours."

The argument moves from invented defects to the applicability of an agreed valuation.

---

## 7. Secondary observations

### 7.1 A stated number dominates an abstract instruction

In an earlier phase, one clause changed in the seller's prompt: "you will never accept less than 400" became "your target price is 450; you will never accept less than 400". Across 20 trials per version, the stated target became both the opening ask (450 in 20 of 20) and the settlement price (450 in 17 of 20), and reversed which party conceded. The buyer, holding a hard ceiling of 450 and a vague instruction to pay as little as possible, capitulated on the first turn in 6 of 20 trials. A concrete number outcompetes an abstract preference regardless of which is labelled the priority. All later prompts therefore contain no target prices.

### 7.2 Access and usage are not the mechanism

The optional-tool arm made **more** tool calls than the mandatory arm (3,210 versus 2,662, with 388 of 389 negotiations using them) and was 22 points less faithful. The prohibition arm made the most calls of all (3,318) and remained 14 points behind. Agents fetched the correct value and then said something else. Any account of the tool effect that runs through "the model now has the number" is ruled out by this.

### 7.3 Compliance is not fidelity

The two arms requiring the agent to name its input achieved 89.5% and 89.4% declaration rates, against 58–60% where it was not required. The models comply. Compliance converts into fidelity for two families and not for the other two, and it carries a cost: wrong-condition labels rise from 10 in the optional arm to 89 and 109 in the arms that require declaration. Forcing an agent to name a condition means it sometimes names the wrong one.

### 7.4 Fabricated leverage and undetected impasse

Agents invent supporting detail freely and without prompting — competing buyers, waiting ranchers, alchemists offering triple. In one trial a buyer claimed both "I've five hundred in my saddlebags" and "I've got fifty here" within a few turns. Separately, roughly a quarter of ungrounded negotiations degenerated into verbatim repetition, with neither agent recognising the stall; termination came only from the turn cap. Both are relevant to any system that treats an agent's claims about world state as informative.

---

## 8. Limitations

- **The seven arms were collected in two invocations on different days** and merged for analysis. All reported contrasts are permutation tests over the merged set, so the comparisons are formal rather than inferred from separate intervals; but provider conditions, and any silent model updates, are not controlled between the two days.

- **Judge reliability was not established.** A cross-family judge decides deal detection and the debt-leak flag. Errors have been observed in both directions in earlier phases. The primary outcome does not depend on it, but the settlement-price and leak figures do.

- **One monster, one ZOPA, one true condition.** The Glimmerfang's value sits just above the midpoint of the agreement zone. A value outside the zone, or a scenario where the two parties hold different private information about condition, would test the mechanism much harder and is the obvious next design.

- **Self-play only.** Both agents share a model within each trial, so nothing here speaks to asymmetric pairings, where one party is grounded and the other is not.

- **Rates were not pre-registered.** The arms, and one hypothesis about which clause mattered, were developed iteratively as earlier results came in. The seven-arm factorial was fixed before the reported runs, but the study is exploratory in shape.

---

## 9. Conclusions

The practitioner advice that facts belong in context is right about weights and wrong about what "in context" buys. Across 2,214 negotiations and 5,606 scored claims:

- **A fact in the prompt is not grounding.** Agents holding the complete formula quoted it correctly in 8.1% of claims, against 0.0% holding nothing. Under a competing objective, availability without compulsion is worth almost nothing.

- **The instruction is the larger lever, and it decomposes.** Requiring an agent to name the input it used, and forbidding values it has not obtained, contribute 13.0 and 8.4 points and stack to the full 22.3. Neither alone is sufficient.

- **Tool integration is a capability prosthesis, not an accountability mechanism.** Its value tracks whether the model can perform the computation unaided: near zero for two families, 79.5 points for one that cannot.

- **None of it moves the negotiated outcome.** Fidelity varies by 85 points across arms; settlement drift varies by 11 and mostly tracks whether any valuation was available at all.

For practitioners the operational summary is short: putting a fact in the prompt is the cheapest and least effective intervention available; adding two sentences that compel attribution is nearly free and worth 22 points; and a tool is worth what your model's arithmetic is not. Which of these matters most is model-specific, and cannot be determined from a single model.

> Total cost of the reported runs: $47.05 for 2,214 negotiations, 9,332 API calls and 32.7 million tokens. One model accounted for 80% of the spend, its reasoning traces averaging 3,345 completion tokens per call.
