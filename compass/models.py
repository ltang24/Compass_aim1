"""The model pool.

Four models, one per GPU, held resident for the whole run. On 6 x 48 GB cards
this leaves two GPUs free; put a fifth agent on one of them if you want, or
leave them for the statistical layer that Aim 1 pairs with this.

The models were picked for two properties. They are all under 10B so they fit on
a single card in bf16 with room for a long context. And they come from different
pretraining lineages, which matters more here than benchmark scores: four
fine-tunes of the same base model would make the same mistakes at the same time,
and a debate between them would look like agreement while telling you nothing.

Swap models by editing config.yaml. Nothing below is specific to a particular
checkpoint.

GPU PLACEMENT
On a shared machine the gpu field in config.yaml goes stale the moment someone
else's job lands on a card. So the transformers backend ignores that field and
places each model, just before loading it, on whichever visible card has the
most free memory right now, one model per card. Cards below MIN_FREE_GB are
skipped rather than crammed, so a card someone is training on will not get
picked, and you get a clear error instead of an OOM. Set COMPASS_MIN_FREE_GB to
override the threshold without editing code.

This uses physical GPU indices. Do not set CUDA_VISIBLE_DEVICES; leave all cards
visible so the picker can see and avoid the busy ones.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelSpec:
    agent_id: str
    model_name: str
    gpu: int
    max_new_tokens: int = 900
    notes: str = ""


# Standard Llama-3 instruct chat template, for checkpoints that ship without one.
LLAMA3_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


DEFAULT_POOL: List[ModelSpec] = [
    ModelSpec("agent_1", "Qwen/Qwen2.5-7B-Instruct", 0,
              notes="strong general reasoning, follows output formats reliably"),
    ModelSpec("agent_2", "meta-llama/Llama-3.1-8B-Instruct", 1,
              notes="gated repo, needs a licence accepted on Hugging Face"),
    ModelSpec("agent_3", "google/gemma-2-9b-it", 2,
              notes="gated repo; different pretraining lineage from the others"),
    ModelSpec("agent_4", "aaditya/Llama3-OpenBioLLM-8B", 3,
              notes="biomedical continued pretraining, adds domain vocabulary"),
    # Optional fifth. Uncomment to use GPU 4.
    # ModelSpec("agent_5", "mistralai/Ministral-8B-Instruct-2410", 4),
]


class Backend:
    def generate(self, agent_id: str, system: str, messages: List[Dict[str, str]],
                 temperature: float, seed: int) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Real backend
# --------------------------------------------------------------------------- #

class TransformersBackend(Backend):
    """One model per GPU, loaded once and kept warm.

    Placement is automatic: each model goes on the visible card with the most
    free memory at the moment it is loaded, ignoring the gpu field in the spec.
    """

    def __init__(self, specs: List[ModelSpec], dtype: str = "bfloat16",
                 trust_remote_code: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.specs = {s.agent_id: s for s in specs}
        self.tok: Dict[str, object] = {}
        self.mdl: Dict[str, object] = {}
        self.placement: Dict[str, int] = {}   # agent_id -> physical gpu index

        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            raise RuntimeError(
                "No GPU visible. Use --backend mock to exercise the pipeline on CPU.")

        min_free_gb = float(os.environ.get("COMPASS_MIN_FREE_GB", "22"))
        taken: set = set()

        def free_gb(i: int) -> float:
            free, _ = torch.cuda.mem_get_info(i)
            return free / 1e9

        def pick_gpu(agent_id: str) -> int:
            """Most-free card not already assigned, above the floor.

            Re-reads memory each call, so the memory a just-loaded model now
            occupies is reflected when the next model is placed.
            """
            best, best_free = None, -1.0
            for i in range(n_gpu):
                if i in taken:
                    continue
                f = free_gb(i)
                if f > best_free:
                    best, best_free = i, f
            if best is None or best_free < min_free_gb:
                raise RuntimeError(
                    f"{agent_id}: no free GPU left with >= {min_free_gb:.0f} GB "
                    f"(best candidate had {best_free:.1f} GB). "
                    f"Free a card, wait for one to clear, or lower the floor with "
                    f"COMPASS_MIN_FREE_GB.")
            taken.add(best)
            return best

        print(f"[load] auto-placement on, floor {min_free_gb:.0f} GB free, "
              f"{n_gpu} cards visible")
        for s in specs:
            dev = pick_gpu(s.agent_id)
            self.placement[s.agent_id] = dev
            print(f"[load] {s.agent_id}: {s.model_name} -> cuda:{dev} "
                  f"({free_gb(dev):.1f} GB free; config said cuda:{s.gpu}, ignored)")
            tok = AutoTokenizer.from_pretrained(
                s.model_name, trust_remote_code=trust_remote_code)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            if tok.chat_template is None:
                # Community checkpoints such as OpenBioLLM ship without a chat
                # template. They are Llama-3 lineage, so borrow the standard
                # Llama-3 instruct template.
                tok.chat_template = LLAMA3_CHAT_TEMPLATE
                print(f"        {s.agent_id}: tokenizer had no chat_template, "
                      f"applied the Llama-3 instruct template")
            mdl = AutoModelForCausalLM.from_pretrained(
                s.model_name,
                torch_dtype=getattr(torch, dtype),
                device_map={"": f"cuda:{dev}"},
                trust_remote_code=trust_remote_code,
            )
            mdl.eval()
            self.tok[s.agent_id] = tok
            self.mdl[s.agent_id] = mdl
            free, total = torch.cuda.mem_get_info(dev)
            print(f"        cuda:{dev} using "
                  f"{(total - free) / 1e9:.1f} GB of {total / 1e9:.1f} GB")

        print(f"[load] placement: "
              + ", ".join(f"{a}=cuda:{d}" for a, d in self.placement.items()))

    def generate(self, agent_id: str, system: str, messages: List[Dict[str, str]],
                 temperature: float, seed: int) -> str:
        torch = self.torch
        spec = self.specs[agent_id]
        tok = self.tok[agent_id]
        mdl = self.mdl[agent_id]

        chat = [{"role": "system", "content": system}] + messages
        try:
            text = tok.apply_chat_template(chat, tokenize=False,
                                           add_generation_prompt=True)
        except Exception:
            # Some chat templates reject a system turn. Fold it into the first user turn.
            merged = [{"role": "user",
                       "content": system + "\n\n" + messages[0]["content"]}]
            merged += messages[1:]
            text = tok.apply_chat_template(merged, tokenize=False,
                                           add_generation_prompt=True)

        enc = tok(text, return_tensors="pt").to(mdl.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = mdl.generate(
                **enc,
                max_new_tokens=spec.max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id,
            )
        return tok.decode(out[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()

    def close(self) -> None:
        self.mdl.clear()
        self.tok.clear()
        gc.collect()
        self.torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #

class MockBackend(Backend):
    """Deterministic stand-in so the pipeline can be exercised without GPUs.

    It reads the verified-facts block out of the prompt and answers from that.
    It is not a model and proves nothing about model behaviour. It exists so you
    can check that loading, retrieval, parsing, verification, scoring and report
    writing all work before you spend GPU hours.
    """

    def __init__(self, specs: List[ModelSpec]):
        self.specs = {s.agent_id: s for s in specs}

    @staticmethod
    def _seeded_choice(agent_id: str, case_key: str, options: List[str]) -> str:
        h = hashlib.md5(f"{agent_id}:{case_key}".encode()).hexdigest()
        return options[int(h, 16) % len(options)]

    def generate(self, agent_id: str, system: str, messages: List[Dict[str, str]],
                 temperature: float, seed: int) -> str:
        prompt = messages[-1]["content"]
        history = " ".join(m["content"] for m in messages)

        hard_stop = "ABSOLUTE CONTRAINDICATION PRESENT" in history
        meets = "Meets the BMI threshold for obesity medicine: yes" in history
        independent = "Separate indication that does not depend on BMI" in history
        case_key = re.search(r"PATIENT RECORD\s+(\S+)", history)
        case_key = case_key.group(1) if case_key else "unknown"

        if "No JSON in this round" in prompt:
            return ("The strongest point against me is that the record may not be "
                    "complete on the contraindication screen. It does not change my "
                    "answer, because the verified facts already resolve the deciding "
                    "question. Nothing a colleague said goes beyond the record.")

        if "Use exactly these headings" in prompt:
            return "MOCK SYNTHESIS. Run with a real backend for usable text."

        if hard_stop:
            initiate, reason = False, "an absolute contraindication is present"
            cards = ["GL-STOP-MTC", "GL-ALT-NONINCRETIN"]
            opts = ["treat the obesity without this drug class",
                    "refer for surgical evaluation", "review again if status changes"]
        elif meets or independent:
            initiate, reason = True, "eligibility is met and no contraindication is present"
            cards = ["GL-ELIG-BMI", "GL-STAGE-INTENSITY"]
            first = self._seeded_choice(agent_id, case_key,
                                        ["semaglutide", "tirzepatide"])
            second = "tirzepatide" if first == "semaglutide" else "semaglutide"
            opts = [first, second, "liraglutide"]
        else:
            initiate, reason = False, "the eligibility threshold is not met"
            cards = ["GL-ELIG-BMI", "GL-NO-PHARM-STAGE1"]
            opts = ["structured lifestyle counselling", "dietitian referral",
                    "recheck in 6 to 12 months"]

        payload = {
            "initiate": initiate,
            "controlling_reason": reason,
            "ranked_options": [
                {"rank": str(i + 1), "option": o, "detail": "mock detail"}
                for i, o in enumerate(opts)
            ],
            "supporting_cards": cards,
            "watch_items": ["mock watch item"],
            "unknowns": ["mock unknown"],
            "confidence": 4,
        }
        return ("Mock reasoning based on the verified facts.\n\n```json\n"
                + json.dumps(payload, indent=2) + "\n```")


# --------------------------------------------------------------------------- #

def build_backend(kind: str, specs: List[ModelSpec], **kw) -> Backend:
    if kind == "mock":
        return MockBackend(specs)
    if kind == "transformers":
        return TransformersBackend(specs, **kw)
    raise ValueError(f"unknown backend: {kind}")


def load_pool(config_path: Optional[str] = None) -> List[ModelSpec]:
    if not config_path or not os.path.exists(config_path):
        return DEFAULT_POOL
    import yaml
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    agents = cfg.get("agents")
    if not agents:
        return DEFAULT_POOL
    return [ModelSpec(agent_id=a["agent_id"], model_name=a["model_name"],
                      gpu=a["gpu"], max_new_tokens=a.get("max_new_tokens", 900),
                      notes=a.get("notes", "")) for a in agents]