#!/usr/bin/env python3
"""Run the COMPASS Aim 1.1 deliberation over the case set.

  python run.py --backend mock                 # no GPU, checks the plumbing
  python run.py --backend transformers         # the real thing
  python run.py --backend transformers --cases CASE-02 CASE-05

Every backend call is recorded to outputs/CASE-XX_transcript.jsonl unless you
pass --no-transcript. That file is the only place the full prompts, the
malformed replies that triggered a REPAIR retry, and the anonymised peer
assignments survive; the in-memory result object keeps none of them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

from compass.deliberation import deliberate
from compass.loader import load_all_patients, load_gold, load_guideline_cards
from compass.models import build_backend, load_pool
from compass.report import build_report, score_case, summarise, write_outputs

ROOT = os.path.dirname(os.path.abspath(__file__))


def _git_commit() -> str:
    """Best effort. The tree may not be a git repo at all."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                   capture_output=True, text=True, timeout=5)
            return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        pass
    return "not a git repo"


def _gpu_snapshot() -> list:
    """What the GPUs looked like when the run started."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return [l.strip() for l in out.stdout.strip().splitlines()]
    except Exception:
        pass
    return []


def _write_run_meta(out_dir: str, args, specs, patients, cards) -> str:
    """Record what was actually run, so a result can be traced back to it."""
    path = os.path.join(out_dir, "run_meta.json")
    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "backend": args.backend,
        "seed": args.seed,
        "dtype": args.dtype if args.backend == "transformers" else None,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
        "gpus": _gpu_snapshot(),
        "agents": [{"agent_id": s.agent_id, "model_name": s.model_name,
                    "gpu": s.gpu} for s in specs],
        "cases": [p.case_id for p in patients],
        "n_cards": len(cards),
        "transcripts": not args.no_transcript,
    }
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="COMPASS Aim 1.1 prototype")
    ap.add_argument("--backend", default="mock", choices=["mock", "transformers"])
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--patients", default=os.path.join(ROOT, "data", "patients"))
    ap.add_argument("--gold", default=os.path.join(ROOT, "data", "gold"))
    ap.add_argument("--cards", default=os.path.join(ROOT, "data", "guidelines",
                                                    "guideline_cards.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--cases", nargs="*", default=None,
                    help="case IDs to run; default is all of them")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--no-score", action="store_true",
                    help="skip scoring against the held-out labels")
    ap.add_argument("--no-transcript", action="store_true",
                    help="do not write the per-call JSONL transcripts")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    patients = load_all_patients(args.patients)
    if args.cases:
        wanted = set(args.cases)
        patients = [p for p in patients if p.case_id in wanted]
        if not patients:
            print(f"No cases matched {sorted(wanted)}", file=sys.stderr)
            return 1
    cards = load_guideline_cards(args.cards)
    specs = load_pool(args.config)

    meta_path = _write_run_meta(args.out, args, specs, patients, cards)

    print(f"COMPASS Aim 1.1 prototype")
    print(f"  backend   {args.backend}")
    print(f"  agents    {len(specs)}")
    for s in specs:
        print(f"            {s.agent_id}  {s.model_name}  (cuda:{s.gpu})")
    print(f"  cases     {len(patients)}")
    print(f"  cards     {len(cards)}")
    print(f"  seed      {args.seed}")
    print(f"  out       {args.out}")
    print(f"  meta      {meta_path}")
    if not args.no_transcript:
        print(f"  transcripts on  ->  {args.out}/CASE-XX_transcript.jsonl")
    print()

    kw = {}
    if args.backend == "transformers":
        kw = {"dtype": args.dtype, "trust_remote_code": args.trust_remote_code}
    backend = build_backend(args.backend, specs, **kw)

    scores = []
    done, failed = [], []
    t0 = time.time()
    try:
        for p in patients:
            print(f"[{p.case_id}] deliberating")
            tc = time.time()
            log_path = None
            if not args.no_transcript:
                log_path = os.path.join(args.out, f"{p.case_id}_transcript.jsonl")
            try:
                result, rubric = deliberate(p, cards, backend, specs,
                                            seed=args.seed,
                                            verbose=not args.quiet,
                                            log_path=log_path)
            except Exception as exc:
                # One bad case should not throw away the cases already done.
                failed.append((p.case_id, repr(exc)))
                print(f"           !! {p.case_id} failed: {exc!r}", file=sys.stderr)
                print()
                continue

            report = build_report(p, rubric, result)
            write_outputs(args.out, p, rubric, result, report)
            done.append(p.case_id)
            print(f"           -> {os.path.join(args.out, p.case_id)}_report.txt")
            if log_path:
                print(f"           -> {log_path}")
            if not args.no_score:
                try:
                    scores.append(score_case(result, load_gold(p.case_id, args.gold)))
                except FileNotFoundError:
                    print(f"           (no gold label for {p.case_id}, not scored)")
            print(f"           {time.time() - tc:.1f}s")
            print()
    finally:
        backend.close()

        # Write whatever we have, even if the run died partway through.
        if scores:
            text = summarise(scores)
            print(text)
            with open(os.path.join(args.out, "scores.json"), "w") as fh:
                json.dump(scores, fh, indent=2)
            with open(os.path.join(args.out, "summary.txt"), "w") as fh:
                fh.write(text + "\n")

        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
            meta.update({
                "finished": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(time.time() - t0, 1),
                "cases_completed": done,
                "cases_failed": failed,
            })
            with open(meta_path, "w") as fh:
                json.dump(meta, fh, indent=2)
        except Exception:
            pass

    print(f"\nfinished in {time.time() - t0:.1f}s")
    print(f"outputs in {args.out}")
    if failed:
        print(f"{len(failed)} case(s) failed: "
              f"{', '.join(c for c, _ in failed)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())