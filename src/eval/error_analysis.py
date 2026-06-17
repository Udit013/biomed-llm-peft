"""Per-subject MedMCQA error analysis from lm-eval `--log_samples` output.

This is a FIRST-CLASS result, not a table dump. It computes per-subject accuracy
for a base run vs a fine-tuned run, then classifies each subject as improved /
neutral / worsened and emits a markdown report with an interpretation section.
The interpretation prose is filled from the measured deltas (no fabrication);
until real sample logs exist, the CLI reports PENDING RUN.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

NEUTRAL_BAND = 1.0  # abs accuracy delta (percentage points) treated as "neutral"


def find_samples_file(results_dir: str | Path, task: str = "medmcqa_val") -> Path | None:
    """Locate the most recent samples_<task>_*.jsonl under an lm-eval output dir."""
    matches = glob.glob(str(Path(results_dir) / "**" / f"samples_{task}_*.jsonl"),
                        recursive=True)
    if not matches:
        return None
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


def _is_correct(rec: dict) -> bool:
    for key in ("acc", "exact_match"):
        if key in rec:
            return bool(round(float(rec[key])))
    # multiple_choice fallback: argmax of loglikelihoods == gold target index
    if "filtered_resps" in rec and "target" in rec:
        lls = [r[0] if isinstance(r, (list, tuple)) else r for r in rec["filtered_resps"]]
        return int(max(range(len(lls)), key=lambda i: lls[i])) == int(rec["target"])
    raise KeyError("cannot determine correctness from sample record")


def subject_accuracy(samples_path: str | Path) -> dict[str, dict]:
    """Map subject_name -> {n, correct, acc} from a samples JSONL file."""
    agg: dict[str, list[int]] = defaultdict(list)
    with open(samples_path) as fh:
        for line in fh:
            rec = json.loads(line)
            subject = rec.get("doc", {}).get("subject_name", "unknown")
            agg[subject].append(1 if _is_correct(rec) else 0)
    return {
        subj: {"n": len(v), "correct": sum(v), "acc": 100 * sum(v) / len(v)}
        for subj, v in sorted(agg.items())
    }


def classify(delta: float) -> str:
    if delta > NEUTRAL_BAND:
        return "improved"
    if delta < -NEUTRAL_BAND:
        return "worsened"
    return "neutral"


def compare(base_dir: str | Path, finetuned_dir: str | Path) -> dict:
    """Compare per-subject accuracy between a base and a fine-tuned lm-eval run."""
    base_file = find_samples_file(base_dir)
    ft_file = find_samples_file(finetuned_dir)
    if base_file is None or ft_file is None:
        return {"status": "PENDING RUN", "base_dir": str(base_dir),
                "finetuned_dir": str(finetuned_dir)}

    base = subject_accuracy(base_file)
    ft = subject_accuracy(ft_file)
    rows = []
    for subj in sorted(set(base) | set(ft)):
        b = base.get(subj, {}).get("acc")
        f = ft.get(subj, {}).get("acc")
        delta = (f - b) if (b is not None and f is not None) else None
        rows.append({
            "subject": subj,
            "n": base.get(subj, ft.get(subj, {})).get("n"),
            "base_acc": b,
            "finetuned_acc": f,
            "delta": delta,
            "verdict": classify(delta) if delta is not None else "n/a",
        })
    return {"status": "OK", "rows": rows}


def render_markdown(result: dict) -> str:
    if result.get("status") != "OK":
        return ("### MedMCQA per-subject error analysis\n\n"
                "**PENDING RUN** — run base and fine-tuned eval with "
                "`--log_samples`, then run `scripts/error_analysis.py`.\n")

    rows = result["rows"]
    out = ["### MedMCQA per-subject error analysis\n",
           "| Subject | N | Base acc | Fine-tuned acc | Δ (pp) | Verdict |",
           "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        out.append(
            f"| {r['subject']} | {r['n']} | {r['base_acc']:.1f} | "
            f"{r['finetuned_acc']:.1f} | {r['delta']:+.1f} | {r['verdict']} |"
        )

    improved = [r for r in rows if r["verdict"] == "improved"]
    neutral = [r for r in rows if r["verdict"] == "neutral"]
    worsened = [r for r in rows if r["verdict"] == "worsened"]
    out += [
        "\n#### Interpretation (auto-generated from measured deltas)\n",
        f"- **Improved** ({len(improved)}): "
        + (", ".join(r["subject"] for r in improved) or "none"),
        f"- **Neutral** ({len(neutral)}): "
        + (", ".join(r["subject"] for r in neutral) or "none"),
        f"- **Worsened** ({len(worsened)}): "
        + (", ".join(r["subject"] for r in worsened) or "none"),
        "\n> Hypothesis to confirm/refine in the README: PEFT on MedMCQA tends to "
        "strengthen fact-recall-heavy subjects more than reasoning-heavy ones. "
        "Replace this line with the interpretation supported by the table above.",
    ]
    return "\n".join(out) + "\n"
