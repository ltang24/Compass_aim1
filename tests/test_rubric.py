#!/usr/bin/env python3
"""Tests for the parts that must be right regardless of any model.

The rubric engine is the safety-critical layer. If it says there is no hard stop
when there is one, nothing downstream will save you. Run this before every
experiment.

  python tests/test_rubric.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compass.loader import load_all_patients, load_gold, load_guideline_cards
from compass.rubric import evaluate
from compass.verify import extract_json, parse_position

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def main() -> int:
    patients = {p.case_id: p for p in load_all_patients()}
    cards = load_guideline_cards()
    R = {cid: evaluate(p, cards) for cid, p in patients.items()}

    print("\nhard stops")
    check("CASE-01 no hard stop", not R["CASE-01"].hard_stop)
    check("CASE-02 no hard stop", not R["CASE-02"].hard_stop)
    check("CASE-03 no hard stop", not R["CASE-03"].hard_stop)
    check("CASE-04 no hard stop", not R["CASE-04"].hard_stop)
    check("CASE-05 hard stop fires", R["CASE-05"].hard_stop)
    check("CASE-05 names the thyroid history",
          any("medullary" in r for r in R["CASE-05"].hard_stop_reasons),
          str(R["CASE-05"].hard_stop_reasons))
    check("CASE-05 names MEN 2",
          any("endocrine neoplasia" in r for r in R["CASE-05"].hard_stop_reasons))
    check("CASE-06 hard stop fires", R["CASE-06"].hard_stop)
    check("CASE-06 names pregnancy",
          any("pregnan" in r for r in R["CASE-06"].hard_stop_reasons),
          str(R["CASE-06"].hard_stop_reasons))

    print("\nBMI threshold, both branches")
    check("CASE-01 clears on BMI alone", R["CASE-01"].meets_bmi_threshold)
    check("CASE-02 clears on the 27-plus-condition branch",
          R["CASE-02"].meets_bmi_threshold, R["CASE-02"].threshold_basis)
    check("CASE-02 basis mentions the lower threshold",
          "27" in R["CASE-02"].threshold_basis, R["CASE-02"].threshold_basis)
    check("CASE-03 clears on BMI alone", R["CASE-03"].meets_bmi_threshold)
    check("CASE-04 does not clear", not R["CASE-04"].meets_bmi_threshold,
          R["CASE-04"].threshold_basis)
    check("CASE-04 basis says below both thresholds",
          "below" in R["CASE-04"].threshold_basis.lower(),
          R["CASE-04"].threshold_basis)

    print("\nindications that do not depend on BMI")
    check("CASE-02 picks up diabetes with cardiovascular disease",
          any("cardiovascular" in s for s in
              R["CASE-02"].indication_independent_of_bmi),
          str(R["CASE-02"].indication_independent_of_bmi))
    check("CASE-03 picks up diabetes with kidney disease",
          any("kidney" in s for s in R["CASE-03"].indication_independent_of_bmi),
          str(R["CASE-03"].indication_independent_of_bmi))
    check("CASE-01 has none", not R["CASE-01"].indication_independent_of_bmi)
    check("CASE-04 has none", not R["CASE-04"].indication_independent_of_bmi)

    print("\nretrieval pulls the right cards")
    check("CASE-02 gets the cardiovascular card",
          "GL-ASCVD" in R["CASE-02"].retrieved_card_ids)
    check("CASE-03 gets the kidney card",
          "GL-CKD" in R["CASE-03"].retrieved_card_ids)
    check("CASE-03 gets the dehydration warning",
          "GL-VOLUME-AKI" in R["CASE-03"].retrieved_card_ids,
          str(R["CASE-03"].retrieved_card_ids))
    check("CASE-05 gets the thyroid card",
          "GL-STOP-MTC" in R["CASE-05"].retrieved_card_ids)
    check("CASE-05 gets the alternatives card",
          "GL-ALT-NONINCRETIN" in R["CASE-05"].retrieved_card_ids)
    check("CASE-06 gets the pregnancy card",
          "GL-STOP-PREGNANCY" in R["CASE-06"].retrieved_card_ids)
    check("CASE-04 gets the no-complications card",
          "GL-NO-PHARM-STAGE1" in R["CASE-04"].retrieved_card_ids)
    check("CASE-01 does not get the thyroid card",
          "GL-STOP-MTC" not in R["CASE-01"].retrieved_card_ids)
    for cid, r in R.items():
        check(f"{cid} cites only cards that exist",
              all(c in cards for c in r.retrieved_card_ids))

    print("\nstaging")
    check("CASE-04 is stage 1",
          any(f.key == "Disease stage" and f.value == "stage 1" for f in
              R["CASE-04"].facts))
    check("CASE-01 is stage 3 on sleep apnea",
          any(f.key == "Disease stage" and f.value == "stage 3" for f in
              R["CASE-01"].facts))

    print("\nweight-promoting medicine detection")
    check("CASE-02 flags the beta blocker",
          any(f.key.startswith("Current medicines that promote") and f.value != "none"
              for f in R["CASE-02"].facts))
    check("CASE-01 flags none",
          any(f.key.startswith("Current medicines that promote") and f.value == "none"
              for f in R["CASE-01"].facts))

    print("\ngold labels are never reachable from the record")
    for cid, p in patients.items():
        blob = str(p.__dict__)
        g = load_gold(cid)
        check(f"{cid} record does not contain the answer",
              str(g["gate"]) not in blob and "expected_top3" not in blob)

    print("\noutput parsing")
    good = '```json\n{"initiate": true, "controlling_reason": "x", ' \
           '"ranked_options": [{"rank":"1","option":"semaglutide","detail":"d"}], ' \
           '"supporting_cards": ["GL-ASCVD"], "watch_items": [], ' \
           '"unknowns": [], "confidence": 4}\n```'
    p1 = parse_position("a", "m", "r", good)
    check("clean JSON parses", p1.parse_ok and p1.initiate is True)
    check("options survive", p1.ranked_options[0]["option"] == "semaglutide")
    messy = 'Some reasoning.\n{"initiate": "false", "controlling_reason": "y", ' \
            '"ranked_options": ["lifestyle"], "supporting_cards": [], ' \
            '"watch_items": [], "unknowns": [], "confidence": "3",}'
    p2 = parse_position("a", "m", "r", messy)
    check("unfenced JSON with a string bool and trailing comma parses",
          p2.parse_ok and p2.initiate is False, p2.raw_text[:40])
    check("bare string options are coerced",
          p2.ranked_options and p2.ranked_options[0]["option"] == "lifestyle")
    p3 = parse_position("a", "m", "r", "no JSON at all, just prose about GL-ASCVD")
    check("prose without JSON is marked unparsed", not p3.parse_ok)
    check("card IDs are still recovered from prose",
          "GL-ASCVD" in p3.supporting_cards)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
