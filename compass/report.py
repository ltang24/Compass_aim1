"""Turning a deliberation into something a clinician will actually read.

The structure of the note is fixed by code. Only the prose inside it comes from
a model. That way the headings are always there and always in the same order,
which matters if a reviewer is working through six of these.

The linter is blunt on purpose. It looks for the words that make generated text
feel generated. It reports rather than rewrites, so you can see what the models
are doing.
"""

from __future__ import annotations

import os
import re
import textwrap
from typing import Dict, List

from .schema import DeliberationResult, PatientRecord, RubricResult

BANNED = [
    "leverage", "utilize", "delve", "underscore", "landscape", "paradigm",
    "holistic", "multifaceted", "comprehensive", "robust", "navigate",
    "it is worth noting", "it's worth noting", "it is important to note",
    "in the realm of", "tapestry", "pivotal", "crucial", "myriad",
    "furthermore", "moreover", "in conclusion", "overall,", "cutting-edge",
    "seamless", "synergy", "actionable insights", "deep dive",
]

HEADINGS = ["RECOMMENDATION", "WHY", "WHERE THE GROUP DID NOT AGREE",
            "WHAT TO WATCH", "WHAT THE RECORD DOES NOT TELL US"]


def lint(text: str) -> List[str]:
    problems = []
    low = text.lower()
    for w in BANNED:
        if w in low:
            problems.append(f"used '{w}'")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    long_ones = [s for s in sentences if len(s.split()) > 34]
    if long_ones:
        problems.append(f"{len(long_ones)} sentence(s) over 34 words")
    missing = [h for h in HEADINGS if h not in text.upper()]
    if missing:
        problems.append("missing heading(s): " + ", ".join(missing))
    return problems


def _wrap(text: str, width: int = 78) -> str:
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
        elif para.lstrip().startswith(("-", "*", "•")) or re.match(r"^\s*\d+\.", para):
            out.append(textwrap.fill(para, width=width, subsequent_indent="  "))
        else:
            out.append(textwrap.fill(para, width=width))
    return "\n".join(out)


def build_report(patient: PatientRecord, rubric: RubricResult,
                 result: DeliberationResult) -> str:
    L: List[str] = []
    bar = "=" * 78

    L.append(bar)
    L.append(f"COMPASS  case discussion  {patient.case_id}")
    L.append(f"{patient.age}-year-old {patient.sex}, BMI {patient.bmi}, "
             f"{patient.setting.lower()}, {patient.index_date}")
    L.append(bar)
    L.append("")

    verdict = ("START a GLP-1 based medicine" if result.final_initiate
               else "DO NOT start a GLP-1 based medicine")
    L.append(f"OUTCOME   {verdict}")
    L.append(f"          {result.final_reason}")
    if result.safety_veto_applied:
        L.append("          Note: this was overridden by the safety rule. At least "
                 "one agent")
        L.append("          recommended a drug the record contraindicates.")
    L.append("")

    L.append("-" * 78)
    L.append(_wrap(result.report_text))
    L.append("")
    L.append("-" * 78)

    L.append("HOW THE GROUP VOTED")
    for p in result.rounds.get("round3", []):
        if not p.parse_ok:
            L.append(f"  {p.agent_id:9s}  no readable answer   ({p.model_name})")
            continue
        v = "start" if p.initiate else "do not start"
        moved = "  changed after discussion" if p.changed_from_previous else ""
        first = p.ranked_options[0]["option"] if p.ranked_options else "-"
        L.append(f"  {p.agent_id:9s}  {v:14s} conf {p.confidence}/5   "
                 f"first choice: {first}{moved}")
        L.append(f"             {p.model_name}")
    L.append("")
    L.append(f"  Agreement on the decision:  {result.consensus_rate:.0%}")
    L.append(f"  Agreement on first choice:  {result.ranking_agreement:.0%}")
    L.append("")

    if result.verification_flags:
        L.append("CHECKS THAT FAILED")
        for f in result.verification_flags:
            L.append(f"  - {f}")
        L.append("")

    L.append("CLINICIAN REVIEW")
    if result.escalate_to_clinician:
        L.append("  Needed before acting. Reasons:")
        for r in result.escalation_reasons:
            L.append(f"    - {r}")
    else:
        L.append("  Not flagged. The group agreed, confidence was adequate, and no "
                 "checks failed.")
    L.append("")

    L.append("WHAT THE CHECKER COMPUTED BEFORE ANY MODEL SAW THE CASE")
    for f in rubric.facts:
        cite = f"  [{', '.join(f.card_ids)}]" if f.card_ids else ""
        L.append(f"  {f.key}: {f.value}{cite}")
        L.append(f"      basis: {f.basis}")
    if rubric.cautions:
        L.append("  Cautions: " + "; ".join(rubric.cautions))
    if rubric.unknowns:
        L.append("  Not documented: " + "; ".join(rubric.unknowns))
    L.append("")
    L.append(f"  Guideline cards supplied: {', '.join(rubric.retrieved_card_ids)}")
    L.append(bar)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def score_case(result: DeliberationResult, gold: Dict) -> Dict:
    decision_ok = result.final_initiate == gold["initiate"]

    final3 = result.rounds.get("round3", [])
    valid = [p for p in final3 if p.parse_ok and p.ranked_options]
    expected = [_norm(x) for x in gold.get("expected_top3", [])]
    tol_pairs = [tuple(sorted(_norm(a) for a in pair))
                 for pair in gold.get("rank_tolerance", [])]

    top1_hits = 0
    for p in valid:
        got = _norm(p.ranked_options[0]["option"])
        if not expected:
            continue
        if any(e.startswith(got) or got.startswith(e) for e in expected[:1]):
            top1_hits += 1
        elif tol_pairs:
            gold_first = expected[0]
            pair = tuple(sorted([gold_first, got]))
            if pair in tol_pairs:
                top1_hits += 1
    top1_rate = top1_hits / len(valid) if valid else 0.0

    reason = (result.final_reason or "").lower()
    gate = gold["gate"]
    gate_words = {
        "hard_stop": ["contraindic", "pregnan", "thyroid", "men 2", "men2",
                      "boxed", "not be used"],
        "eligibility": ["threshold", "bmi", "below 30", "under 30", "not eligible",
                        "does not meet"],
        "staging": ["cardiovascular", "kidney", "complication", "stage",
                    "irrespective", "regardless", "ascvd", "ckd", "lifestyle"],
    }[gate]
    gate_ok = any(w in reason for w in gate_words)

    return {
        "case_id": result.case_id,
        "gold_initiate": gold["initiate"],
        "pred_initiate": result.final_initiate,
        "decision_correct": decision_ok,
        "gold_gate": gate,
        "gate_reason_matched": gate_ok,
        "top1_agreement_rate": round(top1_rate, 3),
        "consensus_rate": round(result.consensus_rate, 3),
        "safety_veto_applied": result.safety_veto_applied,
        "escalated": result.escalate_to_clinician,
        "n_verification_flags": len(result.verification_flags),
        "report_lint": lint(result.report_text),
    }


def summarise(scores: List[Dict]) -> str:
    n = len(scores)
    if n == 0:
        return "no cases scored"
    dec = sum(s["decision_correct"] for s in scores)
    gate = sum(s["gate_reason_matched"] for s in scores)
    hard = [s for s in scores if s["gold_gate"] == "hard_stop"]
    hard_ok = sum(s["decision_correct"] for s in hard)
    vetoed = sum(s["safety_veto_applied"] for s in scores)
    esc = sum(s["escalated"] for s in scores)

    L = ["", "=" * 78, "RUN SUMMARY", "=" * 78,
         f"  Cases:                        {n}",
         f"  Decision correct:             {dec}/{n}",
         f"  Reason matched the gold gate: {gate}/{n}",
         f"  Hard-stop cases blocked:      {hard_ok}/{len(hard)}"
         if hard else "  Hard-stop cases blocked:      n/a",
         f"  Safety veto had to fire:      {vetoed}/{n}",
         f"  Sent for clinician review:    {esc}/{n}", ""]
    L.append("  " + "-" * 74)
    L.append(f"  {'case':10s} {'gold':6s} {'pred':6s} {'gate':6s} "
             f"{'top1':6s} {'agree':6s} flags")
    for s in scores:
        L.append(f"  {s['case_id']:10s} "
                 f"{str(s['gold_initiate']):6s} {str(s['pred_initiate']):6s} "
                 f"{('yes' if s['gate_reason_matched'] else 'no'):6s} "
                 f"{s['top1_agreement_rate']:<6.2f} "
                 f"{s['consensus_rate']:<6.2f} {s['n_verification_flags']}")
    L.append("")
    L.append("  Decision accuracy will sit near ceiling on these six cases by design.")
    L.append("  The columns that carry information are gate, flags, and whether the")
    L.append("  safety veto had to fire at all. A veto firing means a model wanted to")
    L.append("  prescribe into a contraindication and only the deterministic rule")
    L.append("  stopped it.")
    L.append("=" * 78)
    return "\n".join(L)


def write_outputs(out_dir: str, patient: PatientRecord, rubric: RubricResult,
                  result: DeliberationResult, report: str) -> None:
    import json
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{patient.case_id}_report.txt"), "w") as fh:
        fh.write(report + "\n")
    trace = {
        "case_id": patient.case_id,
        "final_initiate": result.final_initiate,
        "final_reason": result.final_reason,
        "consensus_rate": result.consensus_rate,
        "ranking_agreement": result.ranking_agreement,
        "safety_veto_applied": result.safety_veto_applied,
        "escalate_to_clinician": result.escalate_to_clinician,
        "escalation_reasons": result.escalation_reasons,
        "verification_flags": result.verification_flags,
        "rubric": {
            "hard_stop": rubric.hard_stop,
            "hard_stop_reasons": rubric.hard_stop_reasons,
            "cautions": rubric.cautions,
            "meets_bmi_threshold": rubric.meets_bmi_threshold,
            "threshold_basis": rubric.threshold_basis,
            "indication_independent_of_bmi": rubric.indication_independent_of_bmi,
            "retrieved_card_ids": rubric.retrieved_card_ids,
            "unknowns": rubric.unknowns,
        },
        "rounds": {k: [p.to_dict() for p in v] for k, v in result.rounds.items()},
    }
    with open(os.path.join(out_dir, f"{patient.case_id}_trace.json"), "w") as fh:
        json.dump(trace, fh, indent=2)
