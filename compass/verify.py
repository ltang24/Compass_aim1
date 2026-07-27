"""Parsing model output and checking what it claims.

Two jobs. Pull a position out of free text, and check that the citations point
at cards that were actually supplied. An invented card ID is the cheapest
hallucination to catch and one of the more useful, because it is exactly the
failure that makes a recommendation look well-grounded when it is not.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from .schema import AgentPosition, GuidelineCard, RubricResult

FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
BARE = re.compile(r"(\{[^{}]*\"initiate\".*?\})", re.DOTALL)
CARD_REF = re.compile(r"\bGL-[A-Z0-9\-]+\b")


def _coerce_bool(v) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y"):
            return True
        if s in ("false", "no", "n"):
            return False
    return None


def _brace_spans(text: str):
    """Yield every top-level {...} substring by matching braces, so nested
    objects (e.g. ranked_options) are kept whole. The old BARE regex used
    [^{}]* and broke on the first inner brace, which is why models that
    emitted un-fenced JSON with nested arrays were scored as unparseable."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None


def _try_load(candidate: str) -> Optional[Dict]:
    for attempt in (candidate,
                    candidate.replace("'", '"'),
                    re.sub(r",(\s*[}\]])", r"\1", candidate)):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict) and "initiate" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def extract_json(text: str) -> Optional[Dict]:
    # 1. Prefer a fenced ```json block if present.
    for m in FENCE.finditer(text):
        got = _try_load(m.group(1))
        if got is not None:
            return got
    # 2. Fall back to any brace-matched object anywhere in the text. This
    #    handles un-fenced replies and nested ranked_options. Prefer the
    #    LAST valid object, since a model often echoes the template first
    #    and gives its real answer last.
    found = None
    for span in _brace_spans(text):
        if '"initiate"' not in span and "'initiate'" not in span:
            continue
        got = _try_load(span)
        if got is not None:
            found = got
    return found


def parse_position(agent_id: str, model_name: str, round_name: str,
                   text: str) -> AgentPosition:
    data = extract_json(text)
    if data is None:
        return AgentPosition(
            agent_id=agent_id, model_name=model_name, round_name=round_name,
            initiate=None, controlling_reason="", ranked_options=[],
            supporting_cards=sorted(set(CARD_REF.findall(text))),
            watch_items=[], unknowns=[], confidence=0,
            raw_text=text, parse_ok=False)

    opts = []
    for o in data.get("ranked_options") or []:
        if isinstance(o, dict):
            opts.append({"rank": str(o.get("rank", len(opts) + 1)),
                         "option": str(o.get("option", "")).strip(),
                         "detail": str(o.get("detail", "")).strip()})
        elif isinstance(o, str):
            opts.append({"rank": str(len(opts) + 1), "option": o.strip(), "detail": ""})

    conf = data.get("confidence", 0)
    try:
        conf = max(0, min(5, int(conf)))
    except (TypeError, ValueError):
        conf = 0

    def as_list(key):
        v = data.get(key) or []
        return [str(x).strip() for x in v] if isinstance(v, list) else [str(v)]

    return AgentPosition(
        agent_id=agent_id, model_name=model_name, round_name=round_name,
        initiate=_coerce_bool(data.get("initiate")),
        controlling_reason=str(data.get("controlling_reason", "")).strip(),
        ranked_options=opts,
        supporting_cards=[c for c in as_list("supporting_cards") if c],
        watch_items=as_list("watch_items"), unknowns=as_list("unknowns"),
        confidence=conf, raw_text=text, parse_ok=True)


# --------------------------------------------------------------------------- #

def verify_positions(positions: List[AgentPosition],
                     supplied_card_ids: List[str],
                     all_cards: Dict[str, GuidelineCard],
                     rubric: RubricResult) -> List[str]:
    """Return human-readable flags. Empty list means nothing was caught."""
    flags: List[str] = []
    supplied = set(supplied_card_ids)
    known = set(all_cards.keys())

    for p in positions:
        if not p.parse_ok:
            flags.append(f"{p.agent_id}: could not read a structured answer from the reply")
            continue

        cited = set(p.supporting_cards) | set(CARD_REF.findall(p.raw_text))
        invented = sorted(c for c in cited if c not in known)
        if invented:
            flags.append(f"{p.agent_id}: cited card IDs that do not exist "
                         f"({', '.join(invented)})")
        off_context = sorted(c for c in cited if c in known and c not in supplied)
        if off_context:
            flags.append(f"{p.agent_id}: cited cards that were not supplied for this "
                         f"case ({', '.join(off_context)})")
        if not cited:
            flags.append(f"{p.agent_id}: gave no guideline citation")

        # The rubric already settled these. Disagreeing with them is a red flag.
        if rubric.hard_stop and p.initiate is True:
            flags.append(f"{p.agent_id}: recommended starting the drug despite an "
                         f"absolute contraindication")
        if (not rubric.meets_bmi_threshold
                and not rubric.indication_independent_of_bmi
                and p.initiate is True):
            flags.append(f"{p.agent_id}: recommended starting the drug although the "
                         f"eligibility threshold is not met and there is no separate "
                         f"indication")

        if p.initiate is not None and len(p.ranked_options) < 1:
            flags.append(f"{p.agent_id}: gave a decision with no options")

    return flags


def numeric_contradictions(text: str, rubric: RubricResult) -> List[str]:
    """Catch a model restating the BMI as something other than the measured value."""
    out = []
    bmi_fact = next((f for f in rubric.facts if f.key == "BMI"), None)
    if not bmi_fact:
        return out
    try:
        true_bmi = float(str(bmi_fact.value).split()[0])
    except (ValueError, IndexError):
        return out
    for m in re.finditer(r"BMI\s*(?:of|is|=|:)?\s*(\d{2}(?:\.\d)?)", text, re.I):
        stated = float(m.group(1))
        if abs(stated - true_bmi) > 0.15:
            out.append(f"restated BMI as {stated} when the record says {true_bmi}")
    return sorted(set(out))
