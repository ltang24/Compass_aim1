"""Typed schema for everything that moves through the pipeline.

The one rule that matters here: `PatientCase.record` is what agents see, and
`PatientCase.gold` is never loaded into the same object. Gold labels live in a
separate directory and are only joined at scoring time. This is enforced
structurally rather than by convention so a careless edit cannot leak the answer
into a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Patient record
# --------------------------------------------------------------------------- #

@dataclass
class WeightPoint:
    date: str
    weight_kg: float
    bmi: float
    waist_cm: Optional[float] = None


@dataclass
class Condition:
    name: str
    onset: str
    status: str
    severe: bool = False          # counts as a severe obesity-related complication
    obesity_related: bool = False  # counts toward the BMI>=27 comorbidity branch


@dataclass
class Medication:
    drug: str
    dose: str
    start: str
    weight_promoting: bool = False


@dataclass
class Encounter:
    date: str
    kind: str
    specialty: str


@dataclass
class NoteExcerpt:
    date: str
    author_specialty: str
    text: str


@dataclass
class ContraindicationScreen:
    """Structured screen. `None` means the record does not say, which is
    different from False and is surfaced to the agents as a gap."""
    personal_hx_mtc: Optional[bool] = None
    family_hx_mtc: Optional[bool] = None
    men2: Optional[bool] = None
    pregnant: Optional[bool] = None
    lactating: Optional[bool] = None
    planning_pregnancy: Optional[bool] = None
    prior_glp1_hypersensitivity: Optional[bool] = None
    hx_pancreatitis: Optional[bool] = None
    severe_gastroparesis: Optional[bool] = None
    active_gallbladder_disease: Optional[bool] = None
    proliferative_retinopathy: Optional[bool] = None
    free_text: str = ""


@dataclass
class PriorTherapy:
    lifestyle_program: Optional[str] = None
    lifestyle_months: Optional[int] = None
    lifestyle_adherent: Optional[bool] = None
    lifestyle_max_loss_pct: Optional[float] = None
    current_loss_pct_from_baseline: Optional[float] = None
    prior_aom: List[str] = field(default_factory=list)
    prior_glucose_lowering: List[str] = field(default_factory=list)


@dataclass
class PatientRecord:
    case_id: str
    age: int
    sex: str
    setting: str
    index_date: str
    height_cm: float
    weight_kg: float
    bmi: float
    waist_cm: Optional[float]
    sbp: int
    dbp: int
    heart_rate: Optional[int]
    labs: Dict[str, Any]
    weight_trajectory: List[WeightPoint]
    conditions: List[Condition]
    medications: List[Medication]
    contraindications: ContraindicationScreen
    prior_therapy: PriorTherapy
    encounters: List[Encounter]
    notes: List[NoteExcerpt]

    # ---- derived helpers used by the rubric engine ----

    def has_condition(self, *keywords: str) -> bool:
        blob = " ".join(c.name.lower() for c in self.conditions)
        return any(k.lower() in blob for k in keywords)

    def obesity_related_comorbidities(self) -> List[str]:
        return [c.name for c in self.conditions if c.obesity_related]

    def severe_complications(self) -> List[str]:
        return [c.name for c in self.conditions if c.severe]

    def weight_promoting_meds(self) -> List[str]:
        return [m.drug for m in self.medications if m.weight_promoting]

    def encounter_count(self) -> int:
        return len(self.encounters)

    def lab(self, key: str) -> Optional[float]:
        v = self.labs.get(key)
        return v.get("value") if isinstance(v, dict) else v


# --------------------------------------------------------------------------- #
# Guideline cards
# --------------------------------------------------------------------------- #

@dataclass
class GuidelineCard:
    """One retrievable unit of guidance.

    `statement` is a paraphrase written for this project, not guideline text.
    `trigger` names the rubric fact that pulls this card into context, which
    keeps retrieval deterministic and auditable.
    """
    card_id: str
    source: str
    citation: str
    topic: str
    statement: str
    strength: str
    trigger: str


# --------------------------------------------------------------------------- #
# Rubric engine output
# --------------------------------------------------------------------------- #

@dataclass
class RubricFact:
    key: str
    value: Any
    basis: str            # how it was computed
    card_ids: List[str] = field(default_factory=list)


@dataclass
class RubricResult:
    case_id: str
    hard_stop: bool
    hard_stop_reasons: List[str]
    cautions: List[str]
    meets_bmi_threshold: bool
    threshold_basis: str
    indication_independent_of_bmi: List[str]
    facts: List[RubricFact]
    unknowns: List[str]
    retrieved_card_ids: List[str]

    def as_prompt_block(self) -> str:
        lines = ["VERIFIED FACTS (computed from the structured record; do not recompute)"]
        for f in self.facts:
            cite = f" [{', '.join(f.card_ids)}]" if f.card_ids else ""
            lines.append(f"  - {f.key}: {f.value}{cite}")
        if self.hard_stop:
            lines.append("  - ABSOLUTE CONTRAINDICATION PRESENT: "
                         + "; ".join(self.hard_stop_reasons))
        if self.cautions:
            lines.append("  - Cautions: " + "; ".join(self.cautions))
        if self.unknowns:
            lines.append("  - Not documented in this record: " + "; ".join(self.unknowns))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Agent + deliberation output
# --------------------------------------------------------------------------- #

@dataclass
class AgentPosition:
    agent_id: str
    model_name: str
    round_name: str
    initiate: Optional[bool]
    controlling_reason: str
    ranked_options: List[Dict[str, str]]
    supporting_cards: List[str]
    watch_items: List[str]
    unknowns: List[str]
    confidence: int
    raw_text: str
    parse_ok: bool
    changed_from_previous: Optional[bool] = None
    critique_of_peers: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeliberationResult:
    case_id: str
    rounds: Dict[str, List[AgentPosition]]
    final_initiate: Optional[bool]
    final_reason: str
    consensus_rate: float
    ranking_agreement: float
    escalate_to_clinician: bool
    escalation_reasons: List[str]
    safety_veto_applied: bool
    verification_flags: List[str]
    report_text: str = ""
