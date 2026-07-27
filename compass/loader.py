"""Load patient records and guideline cards, and render a record as prompt text.

Gold labels are loaded by a separate function that the deliberation code never
calls. If you are adding features here, keep it that way.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

from .schema import (Condition, ContraindicationScreen, Encounter, GuidelineCard,
                     Medication, NoteExcerpt, PatientRecord, PriorTherapy, WeightPoint)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Patient records
# --------------------------------------------------------------------------- #

def load_patient(path: str) -> PatientRecord:
    with open(path) as fh:
        d = json.load(fh)
    return PatientRecord(
        case_id=d["case_id"], age=d["age"], sex=d["sex"], setting=d["setting"],
        index_date=d["index_date"], height_cm=d["height_cm"], weight_kg=d["weight_kg"],
        bmi=d["bmi"], waist_cm=d.get("waist_cm"), sbp=d["sbp"], dbp=d["dbp"],
        heart_rate=d.get("heart_rate"), labs=d["labs"],
        weight_trajectory=[WeightPoint(**w) for w in d["weight_trajectory"]],
        conditions=[Condition(**c) for c in d["conditions"]],
        medications=[Medication(**m) for m in d["medications"]],
        contraindications=ContraindicationScreen(**d["contraindications"]),
        prior_therapy=PriorTherapy(**d["prior_therapy"]),
        encounters=[Encounter(**e) for e in d["encounters"]],
        notes=[NoteExcerpt(**n) for n in d["notes"]],
    )


def load_all_patients(data_dir: str | None = None) -> List[PatientRecord]:
    data_dir = data_dir or os.path.join(_repo_root(), "data", "patients")
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(
            f"No patient files in {data_dir}. Run: python data/build_dataset.py")
    return [load_patient(f) for f in files]


def load_gold(case_id: str, gold_dir: str | None = None) -> Dict:
    """Only the scorer calls this. Never call it from prompt-building code."""
    gold_dir = gold_dir or os.path.join(_repo_root(), "data", "gold")
    with open(os.path.join(gold_dir, f"{case_id}.json")) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Guideline cards
# --------------------------------------------------------------------------- #

def load_guideline_cards(path: str | None = None) -> Dict[str, GuidelineCard]:
    path = path or os.path.join(_repo_root(), "data", "guidelines",
                                "guideline_cards.json")
    with open(path) as fh:
        blob = json.load(fh)
    cards = [GuidelineCard(**c) for c in blob["cards"]]
    return {c.card_id: c for c in cards}


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #

def _fmt_lab(name: str, entry) -> str:
    if not isinstance(entry, dict):
        return f"{name}: {entry}"
    val, unit = entry.get("value"), entry.get("unit", "")
    lo, hi = entry.get("ref_low"), entry.get("ref_high")
    flag = ""
    if isinstance(val, (int, float)):
        if hi is not None and val > hi:
            flag = "  HIGH"
        elif lo is not None and val < lo:
            flag = "  LOW"
    ref = ""
    if lo is not None and hi is not None:
        ref = f" (ref {lo}-{hi})"
    elif hi is not None:
        ref = f" (ref under {hi})"
    elif lo is not None:
        ref = f" (ref over {lo})"
    return f"{name}: {val} {unit}{ref}{flag}".strip()


LAB_LABELS = {
    "hba1c_pct": "HbA1c", "fasting_glucose_mg_dl": "Fasting glucose",
    "ldl_mg_dl": "LDL cholesterol", "hdl_mg_dl": "HDL cholesterol",
    "triglycerides_mg_dl": "Triglycerides", "alt_u_l": "ALT", "ast_u_l": "AST",
    "creatinine_mg_dl": "Creatinine", "egfr": "eGFR", "uacr_mg_g": "Urine ACR",
    "potassium_mmol_l": "Potassium", "tsh_miu_l": "TSH",
    "calcitonin_pg_ml": "Calcitonin", "cea_ng_ml": "CEA",
    "hemoglobin_g_dl": "Hemoglobin", "hcg_qualitative": "Pregnancy test",
}


def render_record(p: PatientRecord) -> str:
    """Plain-text view of the record. This is what every agent reads."""
    out: List[str] = []
    out.append(f"PATIENT RECORD  {p.case_id}")
    out.append(f"{p.age}-year-old {p.sex}. {p.setting}. Visit date {p.index_date}.")
    out.append("")

    out.append("MEASUREMENTS")
    waist = f", waist {p.waist_cm} cm" if p.waist_cm else ""
    out.append(f"  Height {p.height_cm} cm, weight {p.weight_kg} kg, "
               f"BMI {p.bmi} kg/m2{waist}")
    hr = f", heart rate {p.heart_rate}" if p.heart_rate else ""
    out.append(f"  Blood pressure {p.sbp}/{p.dbp}{hr}")
    out.append("")

    out.append("WEIGHT OVER THE LAST 24 MONTHS")
    for w in p.weight_trajectory:
        out.append(f"  {w.date}   {w.weight_kg} kg   BMI {w.bmi}")
    out.append("")

    out.append("LABS")
    for key, entry in p.labs.items():
        out.append("  " + _fmt_lab(LAB_LABELS.get(key, key), entry))
    out.append("")

    out.append("PROBLEM LIST")
    if not p.conditions:
        out.append("  Nothing on the problem list.")
    for c in p.conditions:
        tag = " [severe complication]" if c.severe else ""
        out.append(f"  {c.name} (since {c.onset}) - {c.status}{tag}")
    out.append("")

    out.append("CURRENT MEDICINES")
    for m in p.medications:
        wp = "  [known to promote weight gain]" if m.weight_promoting else ""
        out.append(f"  {m.drug} {m.dose}, started {m.start}{wp}")
    out.append("")

    out.append("CONTRAINDICATION SCREEN")
    c = p.contraindications
    screen = [
        ("Personal history of medullary thyroid carcinoma", c.personal_hx_mtc),
        ("Family history of medullary thyroid carcinoma", c.family_hx_mtc),
        ("MEN 2 syndrome", c.men2),
        ("Pregnant", c.pregnant),
        ("Breastfeeding", c.lactating),
        ("Planning a pregnancy", c.planning_pregnancy),
        ("Previous serious reaction to a GLP-1 medicine", c.prior_glp1_hypersensitivity),
        ("History of pancreatitis", c.hx_pancreatitis),
        ("Severe gastroparesis", c.severe_gastroparesis),
        ("Active gallbladder disease", c.active_gallbladder_disease),
        ("Proliferative diabetic retinopathy", c.proliferative_retinopathy),
    ]
    for label, val in screen:
        mark = "not recorded" if val is None else ("YES" if val else "no")
        out.append(f"  {label}: {mark}")
    if c.free_text:
        out.append(f"  Comment: {c.free_text}")
    out.append("")

    out.append("WEIGHT TREATMENT SO FAR")
    t = p.prior_therapy
    if t.lifestyle_program:
        adh = ("stuck with it" if t.lifestyle_adherent else "attended on and off")
        out.append(f"  {t.lifestyle_program}, {t.lifestyle_months} months, {adh}.")
        if t.lifestyle_max_loss_pct is not None:
            out.append(f"  Best weight loss reached {t.lifestyle_max_loss_pct}% of "
                       f"starting weight; currently "
                       f"{t.current_loss_pct_from_baseline}% below baseline.")
    else:
        out.append("  No structured lifestyle program recorded.")
    out.append(f"  Previous weight medicines: "
               f"{', '.join(t.prior_aom) if t.prior_aom else 'none'}")
    if t.prior_glucose_lowering:
        out.append(f"  Glucose medicines beyond metformin: "
                   f"{', '.join(t.prior_glucose_lowering)}")
    out.append("")

    out.append(f"VISIT HISTORY ({len(p.encounters)} outpatient visits in 24 months)")
    for e in p.encounters:
        out.append(f"  {e.date}  {e.specialty}, {e.kind}")
    out.append("")

    out.append("NOTES")
    for n in p.notes:
        out.append(f"  {n.date}, {n.author_specialty}:")
        out.append(f"    {n.text}")
    return "\n".join(out)


def render_cards(cards: List[GuidelineCard]) -> str:
    out = ["GUIDELINE CARDS AVAILABLE FOR THIS CASE",
           "(cite by card ID; do not cite anything not listed here)"]
    for c in cards:
        out.append("")
        out.append(f"[{c.card_id}]  {c.topic}  ({c.strength})")
        out.append(f"  Source: {c.source}")
        out.append(f"  {c.statement}")
    return "\n".join(out)
