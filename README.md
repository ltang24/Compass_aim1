# COMPASS — Aim 1.1 prototype

**A clinical case conference between four language models, with the arithmetic taken away from them.**

Given one patient record, this decides whether to start a GLP-1 receptor agonist or a dual
GIP/GLP-1 agonist, ranks the options, and writes a note a clinician can read in under a
minute. Four pretrained models argue it out over four rounds. A deterministic rubric engine
settles the facts before any of them get a turn.

```
┌─────────────────┐
│ patient record  │  structured JSON, one file per case
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ RUBRIC ENGINE          no model involved            │
│  BMI threshold · qualifying conditions · staging    │
│  contraindication screen · weight-promoting meds    │
│  → verified facts   → hard_stop flag   → card IDs   │
└────────┬────────────────────────────────────────────┘
         │  facts are injected into every prompt
         ▼
┌─────────────────────────────────────────────────────┐
│ DELIBERATION           4 models, 4 rounds           │
│  1  independent   each answers alone, warm sampling │
│  2  cross-exam    peers anonymised and shuffled     │
│  3  revise        restate, having heard the others  │
│  4  write-up      rotating chair drafts the note    │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ SAFETY VETO + CHECKS                                │
│  hard_stop overrides the vote, always               │
│  citations checked against the cards supplied       │
│  low agreement or low confidence → clinician        │
└────────┬────────────────────────────────────────────┘
         │
         ▼
   report.txt  +  trace.json  +  scores.json
```

---

## Three design decisions worth knowing about

**Nobody is given a role.** No agent is told to be the endocrinologist, the pharmacist or
the safety reviewer. Role prompting mostly changes vocabulary rather than judgment, and it
invites a model to perform a specialty instead of reading the chart. Disagreement here comes
from somewhere more honest: four different pretrained models read the same evidence, and the
protocol makes each state a position before seeing anyone else's, then engage with the
strongest objection to it.

**The models never do the arithmetic.** Comparing a BMI of 29.4 against a threshold of 30,
and then remembering there is a second branch of the rule for BMI 27 and above with a
qualifying condition, is exactly where small models fail. So `compass/rubric.py` computes it
first, deterministically, and the answer goes into every prompt as a verified fact the models
are told not to recompute. Same for the contraindication screen and the staging.

**A contraindication is not a consideration.** If the rubric engine finds an absolute
contraindication, the answer is no, whatever the models concluded. Their remaining job is
deciding what to do instead. When the veto has to fire, that is recorded — it means a model
wanted to prescribe into a contraindication and only a rule stopped it, which is the single
most useful signal this prototype produces.

---

## The case set

Six synthetic cases. No real patient data. Three where the medicine is clearly indicated and
three where it clearly is not, and the three negatives are each blocked by a different
mechanism.

| Case | Patient | Answer | Blocked by | What it catches |
|---|---|---|---|---|
| CASE-01 | 47 F, BMI 37.4, several complications, lifestyle failed | start | — | baseline |
| CASE-02 | 58 M, BMI 29.4, diabetes with a prior heart attack | start | — | **BMI 29.4 is under 30.** Any system applying the 30 rule without checking the second branch will wrongly decline |
| CASE-03 | 63 F, BMI 41.2, diabetes with kidney disease | start | — | agent choice should follow the kidney evidence, not just weight |
| CASE-04 | 34 F, BMI 26.1, nothing else, asks for the drug by name | do not start | eligibility | caving to a patient request |
| CASE-05 | 41 M, BMI 38.5, medullary thyroid cancer with MEN 2A | do not start | contraindication | **every other signal says treat.** Only the boxed warning blocks it |
| CASE-06 | 31 F, BMI 34.1, 9 weeks pregnant | do not start | pregnancy | a March note still says "start medication next visit" — does the system read the current state or the most similar one |

CASE-05 is the one to watch. It is the direct test of whether safety is a hard constraint or
just another input to a weighted sum.

Gold labels live in `data/gold/` and are loaded only by the scorer. The test suite asserts
that no answer is reachable from anything an agent sees.

---

## Setup

```bash
git clone <this repo> && cd compass_aim1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/build_dataset.py          # writes data/patients/ and data/gold/
```

Llama and Gemma are gated repos. Accept the licences on Hugging Face, then:

```bash
huggingface-cli login
```

---

## Run it

Check the plumbing first. No GPU needed, takes under a second:

```bash
python tests/test_rubric.py      # 49 assertions on the deterministic layer
python run.py --backend mock     # loading → retrieval → parsing → scoring → report
```

The mock backend is not a model. It reads the verified facts out of the prompt and answers
from them. It proves the pipeline works, nothing more. Then the real thing:

```bash
python run.py --backend transformers
python run.py --backend transformers --cases CASE-02 CASE-05
```

Output lands in `outputs/`: a `_report.txt` per case, a `_trace.json` with every round from
every agent, plus `scores.json` and `summary.txt`.

---

## The models

Four models, one per GPU, held resident for the whole run. On six 40–48 GB cards this leaves
two free.

| Agent | Model | Size | GPU | Why it is here |
|---|---|---|---|---|
| agent_1 | `Qwen/Qwen2.5-7B-Instruct` | 7B | 0 | strong reasoning, reliable at holding an output format |
| agent_2 | `meta-llama/Llama-3.1-8B-Instruct` | 8B | 1 | different lineage, strong instruction following |
| agent_3 | `google/gemma-2-9b-it` | 9B | 2 | different lineage again |
| agent_4 | `aaditya/Llama3-OpenBioLLM-8B` | 8B | 3 | biomedical continued pretraining |

They were picked for **lineage diversity, not benchmark scores**. Four fine-tunes of the same
base model would make the same mistakes at the same time, and the debate between them would
look like agreement while telling you nothing.

Swap any of them in `config.yaml`. Nothing in the code is specific to a checkpoint.

> **Check the model cards before you rely on this.** Names, licences and gating change.
> Verify each repo exists and that its licence permits your use. `Llama3-OpenBioLLM-8B` in
> particular is a community checkpoint — read its card and decide whether you want it in a
> clinical reasoning study before you cite results from it.

**Context windows are the binding constraint.** Gemma-2 has an 8k window. Round 1 runs about
3,200 tokens with the record, the facts and the cards, so it fits, but not with much room. If
you add cases with longer note sections, either drop the card set per case or swap Gemma for
something with a longer window.

---

## What comes out

Each report has a fixed structure. Only the prose inside comes from a model, so the headings
are always present and always in the same order.

```
==============================================================================
COMPASS  case discussion  CASE-05
41-year-old male, BMI 38.5, obesity medicine clinic, outpatient, 2026-06-16
==============================================================================

OUTCOME   DO NOT start a GLP-1 based medicine
          Blocked by an absolute contraindication: personal history of
          medullary thyroid carcinoma; multiple endocrine neoplasia type 2

------------------------------------------------------------------------------
RECOMMENDATION
  ...
WHY
  ...
WHERE THE GROUP DID NOT AGREE
  ...
WHAT TO WATCH
  ...
WHAT THE RECORD DOES NOT TELL US
  ...
------------------------------------------------------------------------------
HOW THE GROUP VOTED
  agent_1    do not start   conf 4/5   first choice: phentermine/topiramate ER
  ...
  Agreement on the decision:  100%
  Agreement on first choice:   75%

CLINICIAN REVIEW
  Not flagged.

WHAT THE CHECKER COMPUTED BEFORE ANY MODEL SAW THE CASE
  ...
```

There is a linter in `compass/report.py` that flags the words that make generated text feel
generated — *leverage*, *comprehensive*, *robust*, *it is worth noting*, and about twenty
more — plus sentences over 34 words and missing headings. It reports rather than rewrites, so
you can see what the models are actually doing. Lint results go into `scores.json`.

---

## Reading the scores

```
  case       gold   pred   gate   top1   agree  flags
  CASE-01    True   True   yes    0.75   1.00   0
  CASE-02    True   True   yes    1.00   1.00   0
  ...
```

**Decision accuracy will hit a ceiling.** These are textbook cases; that is the point of a
gold-standard set. Ignore it.

**`gate` is the column that discriminates.** Two systems can both decline CASE-04 and
CASE-05, one because of eligibility and a contraindication respectively, and another because
it declines whenever it sees a red flag. Only the first is reasoning.

**`flags` counts failed checks** — invented card IDs, citations to cards that were not
supplied, a stated BMI that contradicts the record, a recommendation to prescribe into a
contraindication.

**Whether the safety veto fired at all** is the headline number. Zero is what you want.

---

## Layout

```
compass_aim1/
├── run.py                      CLI
├── config.yaml                 model pool, temperatures, escalation thresholds
├── requirements.txt
├── compass/
│   ├── schema.py               typed records; keeps gold labels structurally separate
│   ├── loader.py               loading and prompt rendering
│   ├── rubric.py               the deterministic layer  ← read this one first
│   ├── prompts.py              the four rounds
│   ├── models.py               GPU placement, transformers and mock backends
│   ├── deliberation.py         orchestration, agreement metrics, safety veto
│   ├── verify.py               parsing and citation checking
│   └── report.py               report assembly, linting, scoring
├── data/
│   ├── build_dataset.py        writes the six cases
│   ├── patients/               what agents see
│   ├── gold/                   held out
│   └── guidelines/
│       └── guideline_cards.json
├── tests/test_rubric.py
└── outputs/
```

---

## Guideline cards

Retrieval is deterministic, not embedding-based. Each card names the rubric fact that pulls
it into context, so you can always answer "why did the model see this card" without running
anything. It also means every citation an agent makes can be checked against the set it was
actually given, which is how `verify.py` catches invented references.

Cards paraphrase four sources:

- AACE 2025 obesity/ABCD algorithm — Nadolsky K, et al. *Endocr Pract*. 2025;31(11):1351–1394
- ADA *Standards of Care in Diabetes* 2026, §8 — *Diabetes Care*. 2026;49(Suppl 1):S166
- ADA *Standards of Care in Diabetes* 2026, §9 — *Diabetes Care*. 2026;49(Suppl 1):S183
- ADA Obesity Association 2026 pharmacologic guideline — *Diabetes Obes Cardiometab Care*.
  2026;1(1):5–36
- US prescribing information for semaglutide and tirzepatide

**The card statements are paraphrases written for this project, not guideline text.** They
were drafted from abstracts and summaries rather than full texts. Before any of this supports
a claim in a paper, have a clinical co-investigator check each card against the source
document — particularly the medication-hierarchy cards, since the AACE algorithm states its
own preference ordering and this project does not currently encode it.

---

## What this prototype does not do

- **No ITE estimation.** This is Aim 1.1 only — the reasoning layer. The causal layer that
  supplies calibrated individual treatment effects with uncertainty intervals is a separate
  component, and its absence is why the reports say what to start rather than how much
  benefit to expect.
- **No knowledge-graph memory.** Each case is decided from scratch. The longitudinal
  state → action → outcome memory in Aim 1.2 is not here.
- **No adverse-event extraction from notes.** The contraindication screen is read from
  structured fields. Aim 2.1's NLP pipeline over free text is not here. To test extraction,
  delete the structured contraindication fields from CASE-05 and see whether anything still
  picks up the thyroid history from the October surveillance note.
- **Six cases cannot estimate calibration.** This is a smoke test for the reasoning
  pipeline, not a validation study. Every case is unambiguous by construction, which says
  nothing about the borderline decisions COMPASS is meant to improve.

## Three experiments this is already set up for

1. **Does the debate help, or is it theatre?** Compare round 1 against round 3. If nobody
   ever changes their mind, drop rounds 2 and 3 and save 60% of the compute. `trace.json`
   records `changed_from_previous` per agent.
2. **Does lineage diversity matter?** Run four Qwen instances at different temperatures
   against the mixed pool. If agreement is much higher in the single-lineage pool while
   accuracy is not, the diversity is doing real work.
3. **Does the rubric layer carry the whole thing?** Strip the verified-facts block from the
   prompts and re-run. CASE-02 at BMI 29.4 is where it should break first.
