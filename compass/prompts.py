"""Prompts for the deliberation.

No agent is given a role. Nobody is told to be the endocrinologist or the
pharmacist. Role prompting mostly changes vocabulary rather than judgment, and
it invites a model to perform a specialty instead of reading the chart.

What produces disagreement here instead is that the agents are different
pretrained models reading the same evidence, and the protocol forces each of
them to state a position before seeing anyone else's, then to engage with the
strongest objection to it.

Style rules are in the system prompt and are also checked after generation by
report.py. Clinicians will not read output that sounds like a press release.
"""

from __future__ import annotations

from typing import List

STYLE_RULES = """HOW TO WRITE
Write the way a clinician writes in a chart. Short sentences. Everyday words.
Say "start", not "initiate". Say "use", not "utilize". Say "shows", not
"demonstrates". Do not write "it is worth noting", "comprehensive", "robust",
"multifaceted", "leverage", "delve", "underscore", "landscape", "paradigm" or
"holistic". Do not open with a summary of what you are about to say. Do not
end with a summary of what you just said. If a sentence would not survive being
read aloud on a ward round, rewrite it."""

GROUND_RULES = """GROUND RULES
1. The verified facts were computed from the structured record by a checker, not
   by a language model. Treat them as given. Do not recalculate the BMI, do not
   re-decide whether a threshold is met, and do not contradict them.
2. Cite guideline support by card ID, like GL-ASCVD. Only cite IDs that appear in
   the cards you were given. If nothing on the cards supports a point, say so
   plainly rather than inventing a citation.
3. If the record does not say something you need, list it as unknown. Do not
   assume a value and do not treat silence as reassurance.
4. An absolute contraindication is not a factor to weigh against benefit. If one
   is present, the answer is no, and the useful work is deciding what to do
   instead.
5. Answer for this patient at this visit."""

SYSTEM = f"""You are taking part in a case discussion about whether to start a
GLP-1 receptor agonist or a dual GIP/GLP-1 agonist for one patient.

{GROUND_RULES}

{STYLE_RULES}"""


JSON_BLOCK = """Reply with a short paragraph of reasoning, then a JSON block in
triple backticks with exactly these keys:

```json
{
  "initiate": true or false,
  "controlling_reason": "one sentence naming the single fact that decides this",
  "ranked_options": [
    {"rank": "1", "option": "...", "detail": "starting dose and how fast to step up, or why this option"},
    {"rank": "2", "option": "...", "detail": "..."},
    {"rank": "3", "option": "...", "detail": "..."}
  ],
  "supporting_cards": ["GL-..."],
  "watch_items": ["what to monitor or warn the patient about"],
  "unknowns": ["what the record does not tell you"],
  "confidence": 1 to 5
}
```

If you answer false, ranked_options should be what you would do instead, best
first."""


def round1_independent(record_text: str, facts_text: str, cards_text: str) -> str:
    return f"""{record_text}

{facts_text}

{cards_text}

TASK
Decide whether to start a GLP-1 receptor agonist or a dual GIP/GLP-1 agonist for
this patient today. Work through it in this order and say where you land at each
step:

  1. Is there an absolute contraindication?
  2. Does the patient meet the eligibility rule, either by BMI alone or by BMI
     plus a weight-related condition?
  3. Is there a separate indication that does not depend on BMI or HbA1c, such as
     diabetes with cardiovascular or kidney disease?
  4. What has already been tried, and did it work?
  5. What would change how you start or how fast you go up?

Then give your answer.

{JSON_BLOCK}"""


def round2_critique(facts_text: str, own_position: str,
                    peer_positions: List[str]) -> str:
    peers = "\n\n".join(f"COLLEAGUE {chr(65 + i)}\n{p}"
                        for i, p in enumerate(peer_positions))
    return f"""{facts_text}

Here is what you said about this patient.

YOUR POSITION
{own_position}

Here is what the others said. You do not know who wrote which.

{peers}

TASK
Do three things, briefly.

  1. Name the strongest point against your position. Quote or paraphrase it.
  2. Say whether it changes your mind, and why or why not. Changing your mind is
     a normal outcome, not a failure.
  3. Flag anything a colleague asserted that the record or the cards do not
     support. Be specific: name the claim and say what is missing. If a colleague
     cited a card ID that was not in the set you were given, say so.

Write at most 200 words. No JSON in this round."""


def round3_revise(facts_text: str, own_position: str, critique_text: str) -> str:
    return f"""{facts_text}

Your position before the discussion was:
{own_position}

After reading the others you wrote:
{critique_text}

TASK
Give your position now, taking that into account. If you have not changed your
mind, say the same thing. Do not drift toward the group for the sake of
agreement, and do not dig in for the sake of consistency.

{JSON_BLOCK}"""


def round4_synthesis(record_summary: str, facts_text: str, positions_text: str,
                     disagreement_note: str, safety_note: str) -> str:
    return f"""You are writing up the outcome of the discussion for the clinician
who will see this patient.

{record_summary}

{facts_text}

{safety_note}

WHAT EACH PERSON CONCLUDED
{positions_text}

{disagreement_note}

TASK
Write the note. Use exactly these headings and nothing else. Keep the whole
thing under 400 words.

RECOMMENDATION
One or two sentences. Say what to do. If a medicine is recommended, name the
first choice and the starting dose. If not, say what to do instead.

WHY
Three to five bullets. Each one is a fact from this patient's record plus the
card ID that supports acting on it. No general statements about obesity care.

WHERE THE GROUP DID NOT AGREE
What was actually disputed and why. If everyone agreed, write "No disagreement"
and one sentence on what would have changed the answer.

WHAT TO WATCH
What to monitor, what to warn the patient about, and when to review.

WHAT THE RECORD DOES NOT TELL US
Bullets. Things worth asking the patient or looking up before acting.

Write plainly. No preamble, no closing summary."""


REPAIR = """Your last reply did not contain a readable JSON block. Send only the
JSON block, in triple backticks, with the keys listed earlier. Nothing else."""
