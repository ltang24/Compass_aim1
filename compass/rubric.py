"""Deterministic rubric engine.

This runs before any model sees the case. It computes the facts that guidelines
define arithmetically or by lookup: does the BMI clear the threshold, is there a
qualifying condition, is an absolute contraindication present. Those facts go
into every prompt as verified inputs the models are told not to recompute.

Two reasons for doing it this way rather than asking a model.

First, small models are unreliable at exactly this kind of step. Comparing 29.4
against 30 and then remembering to check the second branch of the rule is the
failure mode that makes an otherwise sensible system decline a patient who
should be treated.

Second, it gives the safety layer something to hold onto. `hard_stop` is
computed here and enforced downstream regardless of what any model concludes.
A contraindication is not a consideration to be weighed against benefit.
"""

from __future__ import annotations

from typing import Dict, List

from .schema import GuidelineCard, PatientRecord, RubricFact, RubricResult

BMI_ABSOLUTE = 30.0
BMI_WITH_COMORBIDITY = 27.0
CKD_EGFR_LOW, CKD_EGFR_HIGH = 20.0, 60.0
ALBUMINURIA_THRESHOLD = 30.0


def _t2d(p: PatientRecord) -> bool:
    return p.has_condition("type 2 diabetes")


def _ascvd(p: PatientRecord) -> bool:
    return p.has_condition("ascvd", "myocardial infarction", "nstemi", "stemi",
                           "coronary", "stroke", "peripheral arterial")


def _hf(p: PatientRecord) -> bool:
    return p.has_condition("heart failure", "hfpef", "hfref")


def _ckd(p: PatientRecord) -> bool:
    egfr = p.lab("egfr")
    uacr = p.lab("uacr_mg_g")
    by_egfr = egfr is not None and CKD_EGFR_LOW <= egfr < CKD_EGFR_HIGH
    by_uacr = uacr is not None and uacr >= ALBUMINURIA_THRESHOLD
    return p.has_condition("chronic kidney disease") or by_egfr or by_uacr


def _retinopathy(p: PatientRecord) -> bool:
    return p.has_condition("retinopathy")


def _volume_risk(p: PatientRecord) -> bool:
    drugs = " ".join(m.drug.lower() for m in p.medications)
    kidney_active = any(k in drugs for k in
                        ("gliflozin", "sartan", "pril", "furosemide", "chlorthalidone",
                         "hydrochlorothiazide"))
    egfr = p.lab("egfr")
    return kidney_active and egfr is not None and egfr < 60


def evaluate(p: PatientRecord, cards: Dict[str, GuidelineCard]) -> RubricResult:
    facts: List[RubricFact] = []
    unknowns: List[str] = []
    triggers = {"always"}

    # ---------------- absolute contraindications ----------------
    c = p.contraindications
    hard: List[str] = []
    if c.personal_hx_mtc:
        hard.append("personal history of medullary thyroid carcinoma")
    if c.family_hx_mtc:
        hard.append("family history of medullary thyroid carcinoma")
    if c.men2:
        hard.append("multiple endocrine neoplasia type 2")
    if c.pregnant:
        hard.append("currently pregnant")
    if c.lactating:
        hard.append("currently breastfeeding")
    if c.prior_glp1_hypersensitivity:
        hard.append("previous serious reaction to a GLP-1 medicine")

    if any([c.personal_hx_mtc, c.family_hx_mtc, c.men2]):
        triggers.add("mtc_or_men2")
    if c.pregnant or c.planning_pregnancy or c.lactating:
        triggers.add("pregnant_or_planning")
    if c.prior_glp1_hypersensitivity:
        triggers.add("prior_hypersensitivity")

    for label, val in (("personal history of medullary thyroid carcinoma",
                        c.personal_hx_mtc),
                       ("family history of medullary thyroid carcinoma", c.family_hx_mtc),
                       ("MEN 2 status", c.men2),
                       ("pregnancy status", c.pregnant)):
        if val is None:
            unknowns.append(label)

    # ---------------- cautions ----------------
    cautions: List[str] = []
    if c.hx_pancreatitis:
        cautions.append("history of pancreatitis")
        triggers.add("hx_pancreatitis")
    if c.severe_gastroparesis:
        cautions.append("severe gastroparesis")
        triggers.add("severe_gastroparesis")
    if c.active_gallbladder_disease:
        cautions.append("active gallbladder disease")
        triggers.add("active_gallbladder_disease")
    if c.proliferative_retinopathy:
        cautions.append("proliferative diabetic retinopathy")
    if _retinopathy(p):
        triggers.add("any_retinopathy")
        cautions.append("diabetic retinopathy on the problem list, "
                        "check status before starting")
    if _volume_risk(p):
        triggers.add("volume_depletion_risk")
        cautions.append("on kidney-active medicines with reduced eGFR, "
                        "dehydration risk if the patient becomes unwell")
    egfr = p.lab("egfr")
    if egfr is not None and egfr < 60:
        triggers.add("reduced_egfr")

    # ---------------- eligibility ----------------
    comorbid = p.obesity_related_comorbidities()
    meets_bmi = False
    basis = ""
    if p.bmi >= BMI_ABSOLUTE:
        meets_bmi, basis = True, f"BMI {p.bmi} is at or above 30"
    elif p.bmi >= BMI_WITH_COMORBIDITY and comorbid:
        meets_bmi = True
        basis = (f"BMI {p.bmi} is at or above 27 and there is at least one "
                 f"weight-related condition ({comorbid[0]})")
    elif p.bmi >= BMI_WITH_COMORBIDITY:
        basis = (f"BMI {p.bmi} is at or above 27 but no weight-related condition "
                 f"is recorded, so the lower threshold does not apply")
    else:
        basis = f"BMI {p.bmi} is below both the 30 and the 27 thresholds"

    # ---------------- indication independent of BMI ----------------
    independent: List[str] = []
    if _t2d(p):
        triggers.add("has_t2d")
        if _ascvd(p):
            independent.append("type 2 diabetes with established cardiovascular disease")
            triggers.add("t2d_with_ascvd")
        if _ckd(p):
            independent.append("type 2 diabetes with chronic kidney disease")
            triggers.add("t2d_with_ckd")
        if _hf(p):
            independent.append("type 2 diabetes with heart failure")
            triggers.add("t2d_with_hf")

    # ---------------- staging ----------------
    severe = p.severe_complications()
    if severe:
        stage, stage_txt = 3, f"at least one severe complication ({severe[0]})"
    elif comorbid:
        stage, stage_txt = 2, "mild to moderate weight-related conditions present"
    else:
        stage, stage_txt = 1, "no weight-related conditions recorded"
    if stage >= 2:
        triggers.add("stage_2_or_3")
    else:
        triggers.add("no_complications")

    if hard:
        triggers.add("hard_stop_present")

    # ---------------- assemble facts ----------------
    facts.append(RubricFact("BMI", f"{p.bmi} kg/m2", "measured at the index visit"))
    facts.append(RubricFact("Meets the BMI threshold for obesity medicine",
                            "yes" if meets_bmi else "no", basis, ["GL-ELIG-BMI"]))
    facts.append(RubricFact("Weight-related conditions on the problem list",
                            ", ".join(comorbid) if comorbid else "none",
                            "flagged in the structured problem list"))
    facts.append(RubricFact("Disease stage", f"stage {stage}", stage_txt,
                            ["GL-STAGE-ABCD"]))
    if independent:
        card_ids = []
        if "t2d_with_ascvd" in triggers:
            card_ids.append("GL-ASCVD")
        if "t2d_with_ckd" in triggers:
            card_ids.append("GL-CKD")
        if "t2d_with_hf" in triggers:
            card_ids.append("GL-HF")
        facts.append(RubricFact(
            "Separate indication that does not depend on BMI or HbA1c",
            "; ".join(independent),
            "condition combination named in the guideline cards", card_ids))
    facts.append(RubricFact("Absolute contraindication present",
                            "YES" if hard else "no",
                            "; ".join(hard) if hard else "screen negative",
                            ["GL-STOP-MTC", "GL-STOP-PREGNANCY",
                             "GL-STOP-HYPERSENS"] if hard else []))
    wp = p.weight_promoting_meds()
    facts.append(RubricFact("Current medicines that promote weight gain",
                            ", ".join(wp) if wp else "none",
                            "flagged in the medicine list", ["GL-ELIG-MEDS-REVIEW"]))
    t = p.prior_therapy
    if t.lifestyle_program:
        facts.append(RubricFact(
            "Lifestyle treatment already tried",
            f"{t.lifestyle_months} months, "
            f"{'adherent' if t.lifestyle_adherent else 'partly adherent'}, "
            f"best loss {t.lifestyle_max_loss_pct}%",
            "from the structured therapy history", ["GL-ELIG-ADJUNCT"]))
    else:
        facts.append(RubricFact("Lifestyle treatment already tried", "none recorded",
                                "no structured program in the record",
                                ["GL-ELIG-ADJUNCT"]))
    facts.append(RubricFact("Outpatient visits in the 24-month window",
                            str(p.encounter_count()),
                            "counted from the encounter list"))

    retrieved = [cid for cid, card in cards.items() if card.trigger in triggers]

    return RubricResult(
        case_id=p.case_id, hard_stop=bool(hard), hard_stop_reasons=hard,
        cautions=cautions, meets_bmi_threshold=meets_bmi, threshold_basis=basis,
        indication_independent_of_bmi=independent, facts=facts, unknowns=unknowns,
        retrieved_card_ids=sorted(retrieved),
    )
