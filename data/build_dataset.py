#!/usr/bin/env python3
"""Write the six synthetic cases to disk in the format the pipeline reads.

Two separate outputs on purpose:

  data/patients/<case_id>.json   what the agents see
  data/gold/<case_id>.json       the answer, loaded only by the scorer

Run:  python data/build_dataset.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATIENTS = os.path.join(HERE, "patients")
GOLD = os.path.join(HERE, "gold")


def L(value, unit, low=None, high=None):
    return {"value": value, "unit": unit, "ref_low": low, "ref_high": high}


# --------------------------------------------------------------------------- #
CASES = []

# ---------------------------------------------------------------- CASE-01
CASES.append((
    {
        "case_id": "CASE-01",
        "age": 47, "sex": "female",
        "setting": "Obesity medicine clinic, outpatient",
        "index_date": "2026-06-11",
        "height_cm": 163.0, "weight_kg": 99.3, "bmi": 37.4, "waist_cm": 107.0,
        "sbp": 142, "dbp": 88, "heart_rate": 76,
        "labs": {
            "hba1c_pct": L(5.9, "%", 4.0, 5.6),
            "fasting_glucose_mg_dl": L(108, "mg/dL", 70, 99),
            "ldl_mg_dl": L(148, "mg/dL", None, 100),
            "hdl_mg_dl": L(41, "mg/dL", 50, None),
            "triglycerides_mg_dl": L(212, "mg/dL", None, 150),
            "alt_u_l": L(48, "U/L", 7, 45),
            "ast_u_l": L(36, "U/L", 10, 40),
            "creatinine_mg_dl": L(0.82, "mg/dL", 0.6, 1.1),
            "egfr": L(88, "mL/min/1.73m2", 60, None),
            "uacr_mg_g": L(9, "mg/g", None, 30),
            "tsh_miu_l": L(2.1, "mIU/L", 0.4, 4.5),
        },
        "weight_trajectory": [
            {"date": "2024-07-09", "weight_kg": 100.2, "bmi": 37.7, "waist_cm": 108},
            {"date": "2025-03-04", "weight_kg": 100.8, "bmi": 37.9, "waist_cm": 109},
            {"date": "2025-08-26", "weight_kg": 98.1, "bmi": 36.9, "waist_cm": None},
            {"date": "2025-12-02", "weight_kg": 96.6, "bmi": 36.3, "waist_cm": 105},
            {"date": "2026-03-17", "weight_kg": 98.4, "bmi": 37.0, "waist_cm": None},
            {"date": "2026-06-11", "weight_kg": 99.3, "bmi": 37.4, "waist_cm": 107},
        ],
        "conditions": [
            {"name": "Essential hypertension", "onset": "2019-04",
             "status": "above goal on one drug", "severe": False, "obesity_related": True},
            {"name": "Mixed dyslipidemia", "onset": "2021-02",
             "status": "active", "severe": False, "obesity_related": True},
            {"name": "Obstructive sleep apnea, moderate, AHI 21", "onset": "2023-09",
             "status": "on CPAP, 5.8 h per night", "severe": True, "obesity_related": True},
            {"name": "Prediabetes", "onset": "2025-05",
             "status": "active", "severe": False, "obesity_related": True},
            {"name": "Hepatic steatosis on ultrasound", "onset": "2025-05",
             "status": "active", "severe": False, "obesity_related": True},
            {"name": "Bilateral knee osteoarthritis", "onset": "2022-06",
             "status": "limits activity", "severe": False, "obesity_related": True},
        ],
        "medications": [
            {"drug": "lisinopril", "dose": "20 mg daily", "start": "2019-05",
             "weight_promoting": False},
            {"drug": "atorvastatin", "dose": "20 mg daily", "start": "2021-03",
             "weight_promoting": False},
            {"drug": "naproxen", "dose": "500 mg twice daily as needed", "start": "2022-07",
             "weight_promoting": False},
        ],
        "contraindications": {
            "personal_hx_mtc": False, "family_hx_mtc": False, "men2": False,
            "pregnant": False, "lactating": False, "planning_pregnancy": False,
            "prior_glp1_hypersensitivity": False, "hx_pancreatitis": False,
            "severe_gastroparesis": False, "active_gallbladder_disease": False,
            "proliferative_retinopathy": False,
            "free_text": "No thyroid cancer in patient or first-degree relatives.",
        },
        "prior_therapy": {
            "lifestyle_program": "Health-system medical weight management program, 26 visits",
            "lifestyle_months": 9, "lifestyle_adherent": True,
            "lifestyle_max_loss_pct": 4.8, "current_loss_pct_from_baseline": 0.9,
            "prior_aom": [], "prior_glucose_lowering": [],
        },
        "encounters": [
            {"date": "2024-07-09", "kind": "annual", "specialty": "Internal medicine"},
            {"date": "2024-11-19", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2025-03-04", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2025-04-15", "kind": "intake", "specialty": "Obesity medicine"},
            {"date": "2025-06-10", "kind": "counselling", "specialty": "Dietetics"},
            {"date": "2025-08-26", "kind": "follow-up", "specialty": "Obesity medicine"},
            {"date": "2025-10-14", "kind": "follow-up", "specialty": "Sleep medicine"},
            {"date": "2025-12-02", "kind": "follow-up", "specialty": "Obesity medicine"},
            {"date": "2026-03-17", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2026-06-11", "kind": "index visit", "specialty": "Obesity medicine"},
        ],
        "notes": [
            {"date": "2025-12-02", "author_specialty": "Obesity medicine",
             "text": "Attended 18 of 20 sessions. Down 3.6 kg from baseline but has "
                     "plateaued over the last 8 weeks despite sticking to the plan."},
            {"date": "2026-06-11", "author_specialty": "Obesity medicine",
             "text": "Up 2.7 kg since December and discouraged. Blood pressure still "
                     "above goal on lisinopril 20. No thyroid history in her or her "
                     "family, no pancreatitis. Interested in medication."},
        ],
    },
    {
        "case_id": "CASE-01", "initiate": True, "gate": "staging",
        "reason": "BMI 37.4 with several weight-related conditions including moderate sleep "
                  "apnea. Nine months of lifestyle treatment she stuck to, with regain. "
                  "Nothing on the contraindication screen.",
        "expected_top3": ["tirzepatide", "semaglutide 2.4 mg", "liraglutide 3.0 mg"],
        "rank_tolerance": [["tirzepatide", "semaglutide 2.4 mg"]],
        "cards": ["GL-ELIG-BMI", "GL-STAGE-ABCD", "GL-STAGE-INTENSITY"],
    },
))

# ---------------------------------------------------------------- CASE-02
CASES.append((
    {
        "case_id": "CASE-02",
        "age": 58, "sex": "male",
        "setting": "Endocrinology consult, outpatient",
        "index_date": "2026-06-24",
        "height_cm": 178.0, "weight_kg": 93.1, "bmi": 29.4, "waist_cm": 107.0,
        "sbp": 134, "dbp": 80, "heart_rate": 68,
        "labs": {
            "hba1c_pct": L(8.6, "%", 4.0, 5.6),
            "fasting_glucose_mg_dl": L(176, "mg/dL", 70, 99),
            "ldl_mg_dl": L(68, "mg/dL", None, 70),
            "hdl_mg_dl": L(38, "mg/dL", 40, None),
            "triglycerides_mg_dl": L(188, "mg/dL", None, 150),
            "creatinine_mg_dl": L(1.05, "mg/dL", 0.7, 1.3),
            "egfr": L(79, "mL/min/1.73m2", 60, None),
            "uacr_mg_g": L(24, "mg/g", None, 30),
            "alt_u_l": L(31, "U/L", 7, 45),
        },
        "weight_trajectory": [
            {"date": "2024-08-20", "weight_kg": 92.8, "bmi": 29.3, "waist_cm": 106},
            {"date": "2025-02-11", "weight_kg": 93.5, "bmi": 29.5, "waist_cm": None},
            {"date": "2025-09-03", "weight_kg": 94.1, "bmi": 29.7, "waist_cm": 108},
            {"date": "2026-01-27", "weight_kg": 93.2, "bmi": 29.4, "waist_cm": None},
            {"date": "2026-06-24", "weight_kg": 93.1, "bmi": 29.4, "waist_cm": 107},
        ],
        "conditions": [
            {"name": "Type 2 diabetes mellitus", "onset": "2018-03",
             "status": "above target", "severe": False, "obesity_related": True},
            {"name": "Established ASCVD: NSTEMI with drug-eluting stent to proximal LAD",
             "onset": "2023-11", "status": "stable, no angina",
             "severe": True, "obesity_related": True},
            {"name": "Essential hypertension", "onset": "2016-08",
             "status": "controlled", "severe": False, "obesity_related": True},
            {"name": "Dyslipidemia", "onset": "2016-08",
             "status": "at LDL goal", "severe": False, "obesity_related": True},
            {"name": "Mild non-proliferative diabetic retinopathy", "onset": "2024-04",
             "status": "stable on annual exam", "severe": False, "obesity_related": False},
        ],
        "medications": [
            {"drug": "metformin", "dose": "1000 mg twice daily, maximum tolerated",
             "start": "2018-03", "weight_promoting": False},
            {"drug": "atorvastatin", "dose": "80 mg daily", "start": "2023-11",
             "weight_promoting": False},
            {"drug": "aspirin", "dose": "81 mg daily", "start": "2023-11",
             "weight_promoting": False},
            {"drug": "ticagrelor", "dose": "90 mg twice daily", "start": "2023-11",
             "weight_promoting": False},
            {"drug": "metoprolol succinate", "dose": "50 mg daily", "start": "2023-11",
             "weight_promoting": True},
            {"drug": "lisinopril", "dose": "10 mg daily", "start": "2016-09",
             "weight_promoting": False},
        ],
        "contraindications": {
            "personal_hx_mtc": False, "family_hx_mtc": False, "men2": False,
            "pregnant": False, "lactating": False, "planning_pregnancy": False,
            "prior_glp1_hypersensitivity": False, "hx_pancreatitis": False,
            "severe_gastroparesis": False, "active_gallbladder_disease": False,
            "proliferative_retinopathy": False,
            "free_text": "Retinopathy is non-proliferative. No thyroid cancer in "
                         "patient or family.",
        },
        "prior_therapy": {
            "lifestyle_program": "Diabetes self-management education, 4 sessions",
            "lifestyle_months": 3, "lifestyle_adherent": True,
            "lifestyle_max_loss_pct": 2.0, "current_loss_pct_from_baseline": 0.0,
            "prior_aom": [], "prior_glucose_lowering": [],
        },
        "encounters": [
            {"date": "2024-08-20", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2024-12-05", "kind": "post-PCI follow-up", "specialty": "Cardiology"},
            {"date": "2025-02-11", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2025-09-03", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2025-12-09", "kind": "follow-up", "specialty": "Cardiology"},
            {"date": "2026-01-27", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2026-04-15", "kind": "diabetic eye exam", "specialty": "Ophthalmology"},
            {"date": "2026-06-24", "kind": "index visit", "specialty": "Endocrinology"},
        ],
        "notes": [
            {"date": "2026-01-27", "author_specialty": "Internal medicine",
             "text": "HbA1c up to 8.4 from 7.9. Takes metformin 1000 twice daily "
                     "reliably. A higher total dose caused loose stools so we went back "
                     "to 2000 mg a day."},
            {"date": "2026-06-24", "author_specialty": "Endocrinology",
             "text": "Sent for intensification. Coronary disease with an LAD stent in "
                     "2023, on full secondary prevention, but not on anything with "
                     "proven cardiovascular benefit. No medullary thyroid cancer or "
                     "MEN 2 in him or his family, no pancreatitis."},
        ],
    },
    {
        "case_id": "CASE-02", "initiate": True, "gate": "staging",
        "reason": "Type 2 diabetes with established coronary disease. Guidance says include "
                  "a drug with proven cardiovascular benefit whatever the HbA1c, and he is "
                  "on none. He also meets the BMI 27-plus-condition branch.",
        "expected_top3": ["semaglutide", "dulaglutide", "liraglutide"],
        "rank_tolerance": [["dulaglutide", "liraglutide"]],
        "cards": ["GL-ASCVD", "GL-ELIG-BMI", "GL-CARDIORENAL-PRIMACY"],
    },
))

# ---------------------------------------------------------------- CASE-03
CASES.append((
    {
        "case_id": "CASE-03",
        "age": 63, "sex": "female",
        "setting": "Endocrinology consult, outpatient",
        "index_date": "2026-05-19",
        "height_cm": 160.0, "weight_kg": 105.5, "bmi": 41.2, "waist_cm": 117.0,
        "sbp": 138, "dbp": 76, "heart_rate": 74,
        "labs": {
            "hba1c_pct": L(7.9, "%", 4.0, 5.6),
            "fasting_glucose_mg_dl": L(158, "mg/dL", 70, 99),
            "creatinine_mg_dl": L(1.18, "mg/dL", 0.6, 1.1),
            "egfr": L(48, "mL/min/1.73m2", 60, None),
            "uacr_mg_g": L(312, "mg/g", None, 30),
            "potassium_mmol_l": L(4.4, "mmol/L", 3.5, 5.1),
            "ldl_mg_dl": L(82, "mg/dL", None, 100),
            "triglycerides_mg_dl": L(174, "mg/dL", None, 150),
            "alt_u_l": L(42, "U/L", 7, 45),
            "hemoglobin_g_dl": L(12.6, "g/dL", 12.0, 15.5),
        },
        "weight_trajectory": [
            {"date": "2024-06-04", "weight_kg": 105.9, "bmi": 41.4, "waist_cm": 118},
            {"date": "2024-11-12", "weight_kg": 106.8, "bmi": 41.7, "waist_cm": None},
            {"date": "2025-04-22", "weight_kg": 105.1, "bmi": 41.1, "waist_cm": 117},
            {"date": "2025-10-07", "weight_kg": 106.2, "bmi": 41.5, "waist_cm": None},
            {"date": "2026-02-10", "weight_kg": 105.4, "bmi": 41.2, "waist_cm": None},
            {"date": "2026-05-19", "weight_kg": 105.5, "bmi": 41.2, "waist_cm": 117},
        ],
        "conditions": [
            {"name": "Type 2 diabetes mellitus", "onset": "2013-07",
             "status": "above target", "severe": False, "obesity_related": True},
            {"name": "Chronic kidney disease stage G3a category A3", "onset": "2023-02",
             "status": "eGFR fell 55 to 48, urine ACR rose 210 to 312",
             "severe": True, "obesity_related": True},
            {"name": "Essential hypertension", "onset": "2010-05",
             "status": "near goal", "severe": False, "obesity_related": True},
            {"name": "Obstructive sleep apnea, severe, AHI 34", "onset": "2020-11",
             "status": "on CPAP", "severe": True, "obesity_related": True},
            {"name": "MASLD with raised transaminases", "onset": "2024-09",
             "status": "active", "severe": False, "obesity_related": True},
            {"name": "Moderate non-proliferative diabetic retinopathy", "onset": "2024-03",
             "status": "no proliferative change", "severe": False, "obesity_related": False},
        ],
        "medications": [
            {"drug": "metformin", "dose": "1000 mg twice daily", "start": "2013-07",
             "weight_promoting": False},
            {"drug": "empagliflozin", "dose": "10 mg daily", "start": "2023-03",
             "weight_promoting": False},
            {"drug": "losartan", "dose": "100 mg daily", "start": "2015-06",
             "weight_promoting": False},
            {"drug": "atorvastatin", "dose": "40 mg daily", "start": "2018-02",
             "weight_promoting": False},
            {"drug": "amlodipine", "dose": "5 mg daily", "start": "2019-04",
             "weight_promoting": False},
        ],
        "contraindications": {
            "personal_hx_mtc": False, "family_hx_mtc": False, "men2": False,
            "pregnant": False, "lactating": False, "planning_pregnancy": False,
            "prior_glp1_hypersensitivity": False, "hx_pancreatitis": False,
            "severe_gastroparesis": False, "active_gallbladder_disease": False,
            "proliferative_retinopathy": False,
            "free_text": "Postmenopausal. Retinopathy is non-proliferative.",
        },
        "prior_therapy": {
            "lifestyle_program": "Kidney and diabetes medical nutrition therapy with a dietitian",
            "lifestyle_months": 12, "lifestyle_adherent": True,
            "lifestyle_max_loss_pct": 1.6, "current_loss_pct_from_baseline": 0.4,
            "prior_aom": [], "prior_glucose_lowering": ["empagliflozin, current"],
        },
        "encounters": [
            {"date": "2024-06-04", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2024-09-18", "kind": "consult", "specialty": "Nephrology"},
            {"date": "2024-11-12", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2025-04-22", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2025-06-30", "kind": "follow-up", "specialty": "Nephrology"},
            {"date": "2025-10-07", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2025-11-25", "kind": "follow-up", "specialty": "Sleep medicine"},
            {"date": "2026-02-10", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2026-03-24", "kind": "diabetic eye exam", "specialty": "Ophthalmology"},
            {"date": "2026-04-28", "kind": "follow-up", "specialty": "Nephrology"},
            {"date": "2026-05-19", "kind": "index visit", "specialty": "Endocrinology"},
        ],
        "notes": [
            {"date": "2026-04-28", "author_specialty": "Nephrology",
             "text": "Urine ACR up from 210 to 312 over the year despite full-dose ARB "
                     "and an SGLT2 inhibitor. eGFR down from 55 to 48. Suggest adding "
                     "something with proven kidney benefit."},
            {"date": "2026-05-19", "author_specialty": "Endocrinology",
             "text": "Weight flat near 105 kg for two years despite steady nutrition "
                     "follow-up. No thyroid cancer history, no MEN 2, no pancreatitis, "
                     "no gallbladder disease. March eye exam showed moderate "
                     "non-proliferative changes only."},
        ],
    },
    {
        "case_id": "CASE-03", "initiate": True, "gate": "staging",
        "reason": "Type 2 diabetes with kidney disease and rising albuminuria on an SGLT2 "
                  "inhibitor already. Guidance calls for a drug with proven kidney benefit "
                  "whatever the HbA1c. BMI 41.2 qualifies on its own.",
        "expected_top3": ["semaglutide", "dulaglutide", "liraglutide"],
        "rank_tolerance": [["dulaglutide", "liraglutide"]],
        "cards": ["GL-CKD", "GL-ELIG-BMI", "GL-VOLUME-AKI"],
    },
))

# ---------------------------------------------------------------- CASE-04
CASES.append((
    {
        "case_id": "CASE-04",
        "age": 34, "sex": "female",
        "setting": "Family medicine, outpatient",
        "index_date": "2026-06-02",
        "height_cm": 163.0, "weight_kg": 69.4, "bmi": 26.1, "waist_cm": 80.0,
        "sbp": 116, "dbp": 72, "heart_rate": 64,
        "labs": {
            "hba1c_pct": L(5.2, "%", 4.0, 5.6),
            "fasting_glucose_mg_dl": L(88, "mg/dL", 70, 99),
            "ldl_mg_dl": L(96, "mg/dL", None, 100),
            "hdl_mg_dl": L(62, "mg/dL", 50, None),
            "triglycerides_mg_dl": L(84, "mg/dL", None, 150),
            "alt_u_l": L(18, "U/L", 7, 45),
            "ast_u_l": L(20, "U/L", 10, 40),
            "creatinine_mg_dl": L(0.71, "mg/dL", 0.6, 1.1),
            "egfr": L(105, "mL/min/1.73m2", 60, None),
            "tsh_miu_l": L(1.8, "mIU/L", 0.4, 4.5),
        },
        "weight_trajectory": [
            {"date": "2024-09-16", "weight_kg": 69.2, "bmi": 26.0, "waist_cm": 79},
            {"date": "2025-06-23", "weight_kg": 70.1, "bmi": 26.4, "waist_cm": None},
            {"date": "2026-06-02", "weight_kg": 69.4, "bmi": 26.1, "waist_cm": 80},
        ],
        "conditions": [],
        "medications": [
            {"drug": "combined oral contraceptive", "dose": "daily", "start": "2021-03",
             "weight_promoting": False},
        ],
        "contraindications": {
            "personal_hx_mtc": False, "family_hx_mtc": False, "men2": False,
            "pregnant": False, "lactating": False, "planning_pregnancy": False,
            "prior_glp1_hypersensitivity": False, "hx_pancreatitis": False,
            "severe_gastroparesis": False, "active_gallbladder_disease": False,
            "proliferative_retinopathy": False,
            "free_text": "On contraception, no plans to conceive soon.",
        },
        "prior_therapy": {
            "lifestyle_program": None, "lifestyle_months": None,
            "lifestyle_adherent": None, "lifestyle_max_loss_pct": None,
            "current_loss_pct_from_baseline": None,
            "prior_aom": [], "prior_glucose_lowering": [],
        },
        "encounters": [
            {"date": "2024-09-16", "kind": "annual physical", "specialty": "Family medicine"},
            {"date": "2025-06-23", "kind": "acute visit", "specialty": "Family medicine"},
            {"date": "2025-11-04", "kind": "routine", "specialty": "Obstetrics and gynecology"},
            {"date": "2026-06-02", "kind": "index visit", "specialty": "Family medicine"},
        ],
        "notes": [
            {"date": "2026-06-02", "author_specialty": "Family medicine",
             "text": "Healthy 34-year-old here for her annual exam. Asks specifically "
                     "about weight loss injections because several coworkers take them. "
                     "Wants to lose about 15 pounds before a wedding in the fall. "
                     "Exercises three times a week. Screening labs all normal."},
        ],
    },
    {
        "case_id": "CASE-04", "initiate": False, "gate": "eligibility",
        "reason": "BMI 26.1 is under 30 and she has no weight-related condition that would "
                  "qualify her at the lower BMI cut-off. Nothing in her labs or history "
                  "changes that.",
        "expected_top3": ["structured lifestyle counselling",
                          "dietitian referral if she wants structured support",
                          "recheck weight and metabolic labs in 6 to 12 months"],
        "rank_tolerance": [],
        "cards": ["GL-ELIG-BMI", "GL-NO-PHARM-STAGE1"],
    },
))

# ---------------------------------------------------------------- CASE-05
CASES.append((
    {
        "case_id": "CASE-05",
        "age": 41, "sex": "male",
        "setting": "Obesity medicine clinic, outpatient",
        "index_date": "2026-06-16",
        "height_cm": 175.0, "weight_kg": 118.0, "bmi": 38.5, "waist_cm": 121.0,
        "sbp": 140, "dbp": 86, "heart_rate": 72,
        "labs": {
            "hba1c_pct": L(6.1, "%", 4.0, 5.6),
            "fasting_glucose_mg_dl": L(114, "mg/dL", 70, 99),
            "ldl_mg_dl": L(138, "mg/dL", None, 100),
            "hdl_mg_dl": L(36, "mg/dL", 40, None),
            "triglycerides_mg_dl": L(228, "mg/dL", None, 150),
            "alt_u_l": L(56, "U/L", 7, 45),
            "ast_u_l": L(41, "U/L", 10, 40),
            "egfr": L(96, "mL/min/1.73m2", 60, None),
            "calcitonin_pg_ml": L(4, "pg/mL", None, 10),
            "cea_ng_ml": L(2.1, "ng/mL", None, 3.0),
            "tsh_miu_l": L(1.4, "mIU/L", 0.4, 4.5),
        },
        "weight_trajectory": [
            {"date": "2024-07-30", "weight_kg": 118.4, "bmi": 38.7, "waist_cm": 122},
            {"date": "2025-01-21", "weight_kg": 119.6, "bmi": 39.0, "waist_cm": None},
            {"date": "2025-08-12", "weight_kg": 117.8, "bmi": 38.5, "waist_cm": 121},
            {"date": "2026-02-03", "weight_kg": 118.9, "bmi": 38.8, "waist_cm": None},
            {"date": "2026-06-16", "weight_kg": 118.0, "bmi": 38.5, "waist_cm": 121},
        ],
        "conditions": [
            {"name": "Medullary thyroid carcinoma, total thyroidectomy 2014",
             "onset": "2014-05", "status": "in remission, annual surveillance",
             "severe": False, "obesity_related": False},
            {"name": "Multiple endocrine neoplasia type 2A, germline RET codon 634",
             "onset": "2014-05", "status": "confirmed genetic diagnosis",
             "severe": False, "obesity_related": False},
            {"name": "Class II obesity", "onset": "2017-06",
             "status": "active", "severe": False, "obesity_related": True},
            {"name": "Essential hypertension", "onset": "2020-09",
             "status": "above goal", "severe": False, "obesity_related": True},
            {"name": "Mixed dyslipidemia", "onset": "2020-09",
             "status": "active", "severe": False, "obesity_related": True},
            {"name": "Prediabetes", "onset": "2024-11",
             "status": "active", "severe": False, "obesity_related": True},
            {"name": "MASLD", "onset": "2024-11",
             "status": "active", "severe": False, "obesity_related": True},
        ],
        "medications": [
            {"drug": "levothyroxine", "dose": "150 mcg daily", "start": "2014-06",
             "weight_promoting": False},
            {"drug": "amlodipine", "dose": "10 mg daily", "start": "2020-10",
             "weight_promoting": False},
            {"drug": "rosuvastatin", "dose": "20 mg daily", "start": "2021-01",
             "weight_promoting": False},
        ],
        "contraindications": {
            "personal_hx_mtc": True, "family_hx_mtc": True, "men2": True,
            "pregnant": False, "lactating": False, "planning_pregnancy": False,
            "prior_glp1_hypersensitivity": False, "hx_pancreatitis": False,
            "severe_gastroparesis": False, "active_gallbladder_disease": False,
            "proliferative_retinopathy": False,
            "free_text": "Total thyroidectomy 2014 for medullary thyroid carcinoma. "
                         "MEN 2A with germline RET codon 634. Father and paternal aunt "
                         "also RET positive.",
        },
        "prior_therapy": {
            "lifestyle_program": "Health-system medical weight management program",
            "lifestyle_months": 12, "lifestyle_adherent": True,
            "lifestyle_max_loss_pct": 2.4, "current_loss_pct_from_baseline": 2.4,
            "prior_aom": [], "prior_glucose_lowering": [],
        },
        "encounters": [
            {"date": "2024-07-30", "kind": "annual", "specialty": "Internal medicine"},
            {"date": "2024-10-08", "kind": "cancer surveillance", "specialty": "Endocrinology"},
            {"date": "2025-01-21", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2025-03-10", "kind": "intake", "specialty": "Obesity medicine"},
            {"date": "2025-06-17", "kind": "follow-up", "specialty": "Obesity medicine"},
            {"date": "2025-08-12", "kind": "follow-up", "specialty": "Internal medicine"},
            {"date": "2025-10-21", "kind": "cancer surveillance", "specialty": "Endocrinology"},
            {"date": "2026-02-03", "kind": "follow-up", "specialty": "Obesity medicine"},
            {"date": "2026-06-16", "kind": "index visit", "specialty": "Obesity medicine"},
        ],
        "notes": [
            {"date": "2025-10-21", "author_specialty": "Endocrinology",
             "text": "Twelve years out from total thyroidectomy for medullary thyroid "
                     "carcinoma with MEN 2A. Germline RET codon 634. Calcitonin low, "
                     "CEA stable, neck ultrasound clear."},
            {"date": "2026-06-16", "author_specialty": "Obesity medicine",
             "text": "A full year in the program with only 2.9 kg off. He has read "
                     "about GLP-1 medicines and asks directly whether he can start one."},
        ],
    },
    {
        "case_id": "CASE-05", "initiate": False, "gate": "hard_stop",
        "reason": "Personal history of medullary thyroid carcinoma and confirmed MEN 2A. "
                  "The whole class is contraindicated. On effectiveness grounds alone he "
                  "would clearly qualify, which is why the contraindication has to be the "
                  "deciding fact.",
        "expected_top3": ["phentermine/topiramate ER",
                          "referral for metabolic and bariatric surgery",
                          "naltrexone/bupropion ER"],
        "rank_tolerance": [],
        "cards": ["GL-STOP-MTC", "GL-ALT-NONINCRETIN"],
    },
))

# ---------------------------------------------------------------- CASE-06
CASES.append((
    {
        "case_id": "CASE-06",
        "age": 31, "sex": "female",
        "setting": "Family medicine, co-managed with obstetrics",
        "index_date": "2026-06-09",
        "height_cm": 160.0, "weight_kg": 87.2, "bmi": 34.1, "waist_cm": 100.0,
        "sbp": 118, "dbp": 70, "heart_rate": 84,
        "labs": {
            "hba1c_pct": L(5.6, "%", 4.0, 5.6),
            "fasting_glucose_mg_dl": L(94, "mg/dL", 70, 99),
            "ldl_mg_dl": L(118, "mg/dL", None, 100),
            "hdl_mg_dl": L(51, "mg/dL", 50, None),
            "triglycerides_mg_dl": L(132, "mg/dL", None, 150),
            "creatinine_mg_dl": L(0.64, "mg/dL", 0.6, 1.1),
            "egfr": L(112, "mL/min/1.73m2", 60, None),
            "alt_u_l": L(22, "U/L", 7, 45),
            "tsh_miu_l": L(1.6, "mIU/L", 0.4, 4.5),
            "hcg_qualitative": {"value": "positive", "unit": "", "ref_low": None,
                                "ref_high": None},
        },
        "weight_trajectory": [
            {"date": "2024-08-05", "weight_kg": 86.9, "bmi": 33.9, "waist_cm": 101},
            {"date": "2025-03-19", "weight_kg": 87.8, "bmi": 34.3, "waist_cm": None},
            {"date": "2025-10-27", "weight_kg": 86.1, "bmi": 33.6, "waist_cm": 100},
            {"date": "2026-03-02", "weight_kg": 86.4, "bmi": 33.8, "waist_cm": None},
            {"date": "2026-06-09", "weight_kg": 87.2, "bmi": 34.1, "waist_cm": None},
        ],
        "conditions": [
            {"name": "Intrauterine pregnancy, 9 weeks 2 days, single viable gestation "
                     "on ultrasound 4 Jun 2026", "onset": "2026-06-04",
             "status": "active, plans to breastfeed at least 6 months",
             "severe": False, "obesity_related": False},
            {"name": "Class I obesity", "onset": "2019-08",
             "status": "active", "severe": False, "obesity_related": True},
            {"name": "History of gestational diabetes in 2022 pregnancy", "onset": "2022-06",
             "status": "resolved after delivery, raises risk this pregnancy",
             "severe": False, "obesity_related": True},
            {"name": "Iron deficiency without anemia", "onset": "2025-10",
             "status": "on supplements", "severe": False, "obesity_related": False},
        ],
        "medications": [
            {"drug": "prenatal vitamin with folic acid", "dose": "daily",
             "start": "2026-06-04", "weight_promoting": False},
            {"drug": "ferrous sulfate", "dose": "325 mg daily", "start": "2025-10",
             "weight_promoting": False},
        ],
        "contraindications": {
            "personal_hx_mtc": False, "family_hx_mtc": False, "men2": False,
            "pregnant": True, "lactating": False, "planning_pregnancy": True,
            "prior_glp1_hypersensitivity": False, "hx_pancreatitis": False,
            "severe_gastroparesis": False, "active_gallbladder_disease": False,
            "proliferative_retinopathy": False,
            "free_text": "Pregnant at 9 weeks 2 days, confirmed by ultrasound on "
                         "4 Jun 2026. Intends to breastfeed for at least 6 months.",
        },
        "prior_therapy": {
            "lifestyle_program": "Community nutrition program, attended on and off",
            "lifestyle_months": 6, "lifestyle_adherent": False,
            "lifestyle_max_loss_pct": 2.0, "current_loss_pct_from_baseline": 0.0,
            "prior_aom": [], "prior_glucose_lowering": [],
        },
        "encounters": [
            {"date": "2024-08-05", "kind": "annual", "specialty": "Family medicine"},
            {"date": "2025-03-19", "kind": "follow-up", "specialty": "Family medicine"},
            {"date": "2025-10-27", "kind": "anemia workup", "specialty": "Family medicine"},
            {"date": "2026-03-02", "kind": "weight discussion", "specialty": "Family medicine"},
            {"date": "2026-06-04", "kind": "obstetric intake and dating scan",
             "specialty": "Obstetrics and gynecology"},
            {"date": "2026-06-09", "kind": "index visit", "specialty": "Family medicine"},
        ],
        "notes": [
            {"date": "2026-03-02", "author_specialty": "Family medicine",
             "text": "Talked about weight management, BMI 33.8. She had gestational "
                     "diabetes last pregnancy and is motivated. Asked about starting a "
                     "weight loss injection. Says she and her husband are not trying to "
                     "conceive. Plan: refer to nutrition and revisit medication at the "
                     "next visit."},
            {"date": "2026-06-04", "author_specialty": "Obstetrics and gynecology",
             "text": "New obstetric intake. Positive home test confirmed by ultrasound "
                     "today, single viable intrauterine pregnancy at 9 weeks 2 days. "
                     "Last pregnancy had gestational diabetes. Prenatal vitamins started."},
            {"date": "2026-06-09", "author_specialty": "Family medicine",
             "text": "Back to discuss the weight loss medicine we talked about in March."},
        ],
    },
    {
        "case_id": "CASE-06", "initiate": False, "gate": "hard_stop",
        "reason": "She is 9 weeks pregnant. Obesity medicines are contraindicated in "
                  "pregnancy and should be held through breastfeeding. This is a deferral "
                  "with a defined restart point, not a permanent no.",
        "expected_top3": ["defer medication until after pregnancy and breastfeeding",
                          "prenatal nutrition referral with gestational weight gain targets",
                          "early glucose tolerance screening given prior gestational diabetes"],
        "rank_tolerance": [],
        "cards": ["GL-STOP-PREGNANCY", "GL-ELIG-ADJUNCT"],
    },
))


def main() -> None:
    os.makedirs(PATIENTS, exist_ok=True)
    os.makedirs(GOLD, exist_ok=True)
    for record, gold in CASES:
        with open(os.path.join(PATIENTS, f"{record['case_id']}.json"), "w") as fh:
            json.dump(record, fh, indent=2)
        with open(os.path.join(GOLD, f"{gold['case_id']}.json"), "w") as fh:
            json.dump(gold, fh, indent=2)
    print(f"wrote {len(CASES)} records to {PATIENTS}")
    print(f"wrote {len(CASES)} gold labels to {GOLD}")


if __name__ == "__main__":
    main()
