"""The deliberation itself.

Four rounds:

  1  Independent. Every agent answers alone, at a temperature high enough to let
     them differ. Nobody has seen anyone else.
  2  Cross-examination. Each agent reads the others' positions, anonymised and
     shuffled so that no agent can learn to defer to a particular model. It must
     name the strongest objection to its own view and flag anything a colleague
     claimed that the record does not support.
  3  Revision. Each agent restates its position. Movement here is what makes this
     a deliberation rather than a vote.
  4  Write-up. One agent drafts the note. The chair rotates by case so no single
     model's voice dominates the corpus.

The safety veto sits outside all of this. If the rubric engine found an absolute
contraindication, the final answer is no, whatever the agents concluded, and
their remaining job is what to do instead.

TRANSCRIPT LOGGING
Pass log_path= to deliberate() and every call to the backend is written to a
JSONL file as it happens: the full system prompt, the full message list the
agent actually saw, the raw response, the sampling parameters and the wall
time. That includes the malformed replies that trigger a REPAIR retry, which
the in-memory AgentPosition throws away.

Written incrementally, one line per call, so a run that dies on case five still
leaves a usable record of cases one to four.
"""

from __future__ import annotations

import json
import os
import random
import time
import zlib
from datetime import datetime
from typing import Dict, List, Tuple

from .loader import render_cards, render_record
from .prompts import (REPAIR, SYSTEM, round1_independent, round2_critique,
                      round3_revise, round4_synthesis)
from .rubric import evaluate
from .schema import (AgentPosition, DeliberationResult, GuidelineCard, PatientRecord,
                     RubricResult)
from .verify import numeric_contradictions, parse_position, verify_positions


class _LoggingBackend:
    """Transparent wrapper that records every generate() call to JSONL.

    Delegates everything it does not define (close, load, whatever else the
    real backend exposes) straight through to the wrapped object, so this is
    a drop-in replacement.
    """

    def __init__(self, inner, path: str, case_id: str):
        # inner first: __getattr__ reads it, so it must exist before anything
        # else can miss on attribute lookup.
        self.inner = inner
        self.path = path
        self.case_id = case_id
        self.stage = "setup"          # set by deliberate() as rounds progress
        self.n = 0
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        # Truncate on construction so a re-run of the same case does not append
        # to the previous run's transcript.
        open(path, "w").close()

    def _write(self, record: dict) -> None:
        record["ts"] = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def note(self, kind: str, **fields) -> None:
        """Write a non-generation record: run config, chair, repair events."""
        self._write({"type": kind, "case_id": self.case_id,
                     "stage": self.stage, **fields})

    def generate(self, agent_id, system, messages, **kw):
        t0 = time.time()
        out = self.inner.generate(agent_id, system, messages, **kw)
        self.n += 1
        self._write({
            "type": "generate",
            "case_id": self.case_id,
            "seq": self.n,
            "stage": self.stage,
            "agent_id": agent_id,
            "system": system,
            "messages": messages,
            "response": out,
            "params": dict(kw),
            "elapsed_s": round(time.time() - t0, 2),
        })
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _stable_chair_index(case_id: str, n: int) -> int:
    """Deterministic across processes.

    The original used hash(), which Python salts per process unless
    PYTHONHASHSEED is set. Two runs with identical seeds could therefore pick
    different chairs and produce different write-ups. crc32 is stable.
    """
    return zlib.crc32(case_id.encode("utf-8")) % n


def _position_summary(p: AgentPosition) -> str:
    if not p.parse_ok:
        return "(no readable answer)"
    verdict = "start" if p.initiate else "do not start"
    opts = "; ".join(f"{o['rank']}. {o['option']}" for o in p.ranked_options[:3])
    cards = ", ".join(p.supporting_cards[:4]) or "no citation"
    return (f"Answer: {verdict}. Deciding fact: {p.controlling_reason} "
            f"Options: {opts or 'none given'}. Cards: {cards}. "
            f"Confidence {p.confidence}/5.")


def _agreement(positions: List[AgentPosition]) -> Tuple[float, float]:
    valid = [p for p in positions if p.parse_ok and p.initiate is not None]
    if not valid:
        return 0.0, 0.0
    yes = sum(1 for p in valid if p.initiate)
    consensus = max(yes, len(valid) - yes) / len(valid)

    firsts = [p.ranked_options[0]["option"].lower().split()[0]
              for p in valid if p.ranked_options and p.ranked_options[0]["option"]]
    if not firsts:
        return consensus, 0.0
    top = max(set(firsts), key=firsts.count)
    return consensus, firsts.count(top) / len(firsts)


def _majority(positions: List[AgentPosition]) -> Tuple[bool | None, str]:
    valid = [p for p in positions if p.parse_ok and p.initiate is not None]
    if not valid:
        return None, "no agent produced a readable answer"
    yes = [p for p in valid if p.initiate]
    no = [p for p in valid if not p.initiate]
    if len(yes) > len(no):
        winners = yes
    elif len(no) > len(yes):
        winners = no
    else:
        winners = max([yes, no], key=lambda g: sum(p.confidence for p in g))
    best = max(winners, key=lambda p: p.confidence)
    return winners[0].initiate, best.controlling_reason


def deliberate(patient: PatientRecord, cards: Dict[str, GuidelineCard], backend,
               specs, seed: int = 7, temps=(0.75, 0.5, 0.35, 0.2),
               verbose: bool = True,
               log_path: str | None = None) -> Tuple[DeliberationResult, RubricResult]:

    logger = None
    if log_path:
        logger = _LoggingBackend(backend, log_path, patient.case_id)
        backend = logger

    rubric = evaluate(patient, cards)
    supplied = [cards[c] for c in rubric.retrieved_card_ids]
    record_text = render_record(patient)
    facts_text = rubric.as_prompt_block()
    cards_text = render_cards(supplied)

    rounds: Dict[str, List[AgentPosition]] = {}
    flags: List[str] = []
    repairs: List[str] = []

    def say(msg):
        if verbose:
            print(msg)

    if logger:
        logger.note("run_config",
                    seed=seed, temps=list(temps),
                    agents=[{"agent_id": s.agent_id, "model_name": s.model_name}
                            for s in specs],
                    retrieved_card_ids=rubric.retrieved_card_ids,
                    hard_stop=rubric.hard_stop,
                    hard_stop_reasons=rubric.hard_stop_reasons)

    # ---------------- round 1 ----------------
    say(f"  round 1  independent assessment")
    if logger:
        logger.stage = "round1"
    p1_prompt = round1_independent(record_text, facts_text, cards_text)
    r1: List[AgentPosition] = []
    for s in specs:
        txt = backend.generate(s.agent_id, SYSTEM,
                               [{"role": "user", "content": p1_prompt}],
                               temperature=temps[0], seed=seed)
        pos = parse_position(s.agent_id, s.model_name, "round1", txt)
        if not pos.parse_ok:
            repairs.append(f"{s.agent_id} round1")
            if logger:
                logger.note("repair_triggered", agent_id=s.agent_id,
                            reason="round 1 reply had no readable JSON block")
            txt2 = backend.generate(
                s.agent_id, SYSTEM,
                [{"role": "user", "content": p1_prompt},
                 {"role": "assistant", "content": txt},
                 {"role": "user", "content": REPAIR}],
                temperature=0.1, seed=seed)
            pos = parse_position(s.agent_id, s.model_name, "round1", txt2)
            say(f"           {s.agent_id}: needed a repair pass"
                f"{'' if pos.parse_ok else ', still unreadable'}")
        r1.append(pos)
        say(f"           {s.agent_id}: "
            f"{'start' if pos.initiate else 'do not start'} "
            f"(confidence {pos.confidence}/5)")
    rounds["round1"] = r1

    # ---------------- round 2 ----------------
    say(f"  round 2  cross-examination")
    if logger:
        logger.stage = "round2"
    rng = random.Random(seed)
    r2_text: Dict[str, str] = {}
    for s in specs:
        own = next(p for p in r1 if p.agent_id == s.agent_id)
        peer_ids = [p.agent_id for p in r1 if p.agent_id != s.agent_id]
        peers = [_position_summary(p) for p in r1 if p.agent_id != s.agent_id]
        order = list(range(len(peers)))
        rng.shuffle(order)
        peers = [peers[i] for i in order]
        if logger:
            # Who was COLLEAGUE A, B, C for this agent. Unrecoverable later
            # otherwise, and it is the thing you need to trace influence.
            logger.note("peer_assignment", agent_id=s.agent_id,
                        mapping={chr(65 + j): peer_ids[i]
                                 for j, i in enumerate(order)})
        txt = backend.generate(
            s.agent_id, SYSTEM,
            [{"role": "user",
              "content": round2_critique(facts_text,
                                         _position_summary(own), peers)}],
            temperature=temps[1], seed=seed + 1)
        r2_text[s.agent_id] = txt

    # ---------------- round 3 ----------------
    say(f"  round 3  revised positions")
    if logger:
        logger.stage = "round3"
    r3: List[AgentPosition] = []
    for s in specs:
        before = next(p for p in r1 if p.agent_id == s.agent_id)
        r3_prompt = round3_revise(facts_text, _position_summary(before),
                                  r2_text[s.agent_id])
        txt = backend.generate(
            s.agent_id, SYSTEM, [{"role": "user", "content": r3_prompt}],
            temperature=temps[2], seed=seed + 2)
        pos = parse_position(s.agent_id, s.model_name, "round3", txt)
        if not pos.parse_ok:
            repairs.append(f"{s.agent_id} round3")
            if logger:
                logger.note("repair_triggered", agent_id=s.agent_id,
                            reason="round 3 reply had no readable JSON block")
            txt2 = backend.generate(
                s.agent_id, SYSTEM,
                [{"role": "user", "content": r3_prompt},
                 {"role": "assistant", "content": txt},
                 {"role": "user", "content": REPAIR}],
                temperature=0.1, seed=seed + 2)
            pos = parse_position(s.agent_id, s.model_name, "round3", txt2)
            say(f"           {s.agent_id}: needed a repair pass"
                f"{'' if pos.parse_ok else ', still unreadable'}")
        pos.changed_from_previous = (pos.initiate != before.initiate)
        pos.critique_of_peers = r2_text[s.agent_id]
        r3.append(pos)
        moved = "  (changed)" if pos.changed_from_previous else ""
        say(f"           {s.agent_id}: "
            f"{'start' if pos.initiate else 'do not start'}{moved}")
    rounds["round3"] = r3

    flags += verify_positions(r3, rubric.retrieved_card_ids, cards, rubric)
    for p in r3:
        for c in numeric_contradictions(p.raw_text, rubric):
            flags.append(f"{p.agent_id}: {c}")

    consensus, rank_agree = _agreement(r3)
    majority, majority_reason = _majority(r3)

    # ---------------- safety veto ----------------
    veto = False
    if rubric.hard_stop:
        veto = majority is not False
        final = False
        final_reason = ("Blocked by an absolute contraindication: "
                        + "; ".join(rubric.hard_stop_reasons))
    else:
        final = majority
        final_reason = majority_reason

    # ---------------- escalation ----------------
    escalate: List[str] = []
    if consensus < 0.75:
        escalate.append(f"the group split on the decision "
                        f"({consensus:.0%} agreement)")
    mean_conf = (sum(p.confidence for p in r3 if p.parse_ok)
                 / max(1, sum(1 for p in r3 if p.parse_ok)))
    if mean_conf < 3.0:
        escalate.append(f"average confidence was low ({mean_conf:.1f}/5)")
    if flags:
        escalate.append(f"{len(flags)} verification flag(s) were raised")
    if rubric.unknowns:
        escalate.append("the contraindication screen is incomplete")
    if veto:
        escalate.append("at least one agent recommended a contraindicated drug")

    # ---------------- round 4 ----------------
    chair = specs[_stable_chair_index(patient.case_id, len(specs))]
    say(f"  round 4  write-up  (chair: {chair.agent_id})")
    if logger:
        logger.stage = "round4"
        logger.note("chair_selected", agent_id=chair.agent_id,
                    model_name=chair.model_name,
                    method="crc32(case_id) % n_agents")
    positions_text = "\n".join(f"- {p.agent_id}: {_position_summary(p)}" for p in r3)
    if consensus >= 0.999:
        dis = "Everyone reached the same answer."
    else:
        dis = (f"The group did not fully agree: {consensus:.0%} landed on the same "
               f"answer. Say what the disagreement was actually about.")
    if rubric.hard_stop:
        safety = ("SAFETY RULING (fixed, not open to discussion)\n"
                  "  An absolute contraindication is present: "
                  + "; ".join(rubric.hard_stop_reasons)
                  + ".\n  The recommendation must be not to start this class of drug. "
                    "Write up what to do instead.")
    else:
        safety = ("SAFETY RULING\n  No absolute contraindication was found on the "
                  "structured screen.")

    summary = (f"{patient.age}-year-old {patient.sex}, BMI {patient.bmi}, "
               f"seen in {patient.setting.lower()} on {patient.index_date}.")
    report = backend.generate(
        chair.agent_id, SYSTEM,
        [{"role": "user", "content": round4_synthesis(
            summary, facts_text, positions_text, dis, safety)}],
        temperature=temps[3], seed=seed + 3)

    result = DeliberationResult(
        case_id=patient.case_id, rounds=rounds, final_initiate=final,
        final_reason=final_reason, consensus_rate=consensus,
        ranking_agreement=rank_agree, escalate_to_clinician=bool(escalate),
        escalation_reasons=escalate, safety_veto_applied=veto,
        verification_flags=flags, report_text=report.strip())

    if logger:
        logger.stage = "done"
        logger.note("outcome",
                    chair=chair.agent_id,
                    final_initiate=final,
                    final_reason=final_reason,
                    consensus_rate=consensus,
                    ranking_agreement=rank_agree,
                    safety_veto_applied=veto,
                    escalate_to_clinician=bool(escalate),
                    escalation_reasons=escalate,
                    verification_flags=flags,
                    repair_passes=repairs,
                    changed_mind=[p.agent_id for p in r3
                                  if p.changed_from_previous],
                    total_generate_calls=logger.n)
        if repairs:
            say(f"           {len(repairs)} repair pass(es): {', '.join(repairs)}")

    return result, rubric