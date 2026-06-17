#!/usr/bin/env python
"""Structural validation — runs on macOS/CPU with NO GPU and NO model download.

Checks that the pipeline is wired correctly end to end without producing any
metrics (everything stays PENDING RUN):
  * every src module imports
  * all YAML configs load + inherit correctly
  * the canonical MCQ prompt format renders
  * lm-eval task YAMLs reference resolvable functions
  * the eval command builds
  * results / cost / error-analysis tables render with PENDING placeholders

Exit code 0 == structurally sound. This does NOT train or evaluate anything.
"""
from __future__ import annotations

import importlib

import _bootstrap  # noqa: F401


def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"  ✓ {name}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ {name}: {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    ok = True

    print("[1] importing src modules")
    for mod in [
        "src.utils.config", "src.utils.env", "src.utils.tracking",
        "src.utils.model_card", "src.data.format", "src.data.medmcqa",
        "src.train.sft", "src.eval.harness", "src.eval.error_analysis",
        "src.eval.results", "src.serve.loader", "src.serve.score",
        "src.serve.cost", "src.serve.api",
    ]:
        ok &= check(mod, lambda m=mod: importlib.import_module(m))

    print("[2] loading + inheriting configs")
    from src.utils.config import load_config

    def _cfg():
        for c in ["base", "qlora_5k", "qlora_20k", "qlora_50k", "qlora_full"]:
            cfg = load_config(f"configs/{c}.yaml")
            assert cfg.model.base_model == "Qwen/Qwen2.5-7B-Instruct"
        five = load_config("configs/qlora_5k.yaml")
        assert five.data.train_size == 5000
        assert load_config("configs/qlora_full.yaml").data.train_size == -1
    ok &= check("configs", _cfg)

    print("[3] canonical prompt format")
    from src.data.format import render_question, render_target

    def _fmt():
        q = render_question("Scurvy is caused by deficiency of?",
                            ["A", "Vitamin C", "D", "K"])
        assert "A. A" in q and q.endswith("Answer:")
        assert render_target(1, ["A", "Vitamin C", "D", "K"]) == "B. Vitamin C"
    ok &= check("format", _fmt)

    print("[4] lm-eval task utils resolve")
    import sys
    sys.path.insert(0, "lm_eval_tasks")

    def _tasks():
        import utils as tu  # noqa
        doc = {"question": "q?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 2}
        assert tu.medmcqa_doc_to_text(doc).startswith("Question: q?")
        assert len(tu.medmcqa_doc_to_choice(doc)) == 4
        pdoc = {"question": "q?", "context": {"contexts": ["x", "y"]},
                "final_decision": "yes"}
        assert "Abstract: x y" in tu.pubmedqa_doc_to_text(pdoc)
        assert tu.pubmedqa_doc_to_target(pdoc) == 0
    ok &= check("lm_eval_tasks/utils", _tasks)

    print("[5] eval command builds")
    from src.eval.harness import build_command

    def _cmd():
        cmd = build_command("Qwen/Qwen2.5-7B-Instruct", "results/x",
                            adapter_dir="outputs/qlora_5k", num_fewshot=5)
        assert "lm_eval" in cmd and "peft=outputs/qlora_5k" in ",".join(cmd)
        assert "--apply_chat_template" in cmd
    ok &= check("build_command", _cmd)

    print("[6] tables render with PENDING placeholders")
    from src.eval.results import render_table
    from src.eval.error_analysis import render_markdown
    from src.serve.cost import render_cost_table

    def _tables():
        assert "PENDING RUN" in render_table()
        assert "PENDING RUN" in render_markdown({"status": "PENDING RUN"})
        assert "PENDING RUN" in render_cost_table("results/does_not_exist.json")
    ok &= check("tables", _tables)

    print("[7] tracking degrades without W&B key")
    import os
    from src.utils.tracking import wandb_enabled

    def _track():
        os.environ.pop("WANDB_API_KEY", None)
        assert wandb_enabled() is False
    ok &= check("tracking", _track)

    print("\n" + ("ALL STRUCTURAL CHECKS PASSED ✓" if ok else "STRUCTURAL CHECKS FAILED ✗"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
