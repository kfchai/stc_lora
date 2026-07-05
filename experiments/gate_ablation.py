"""Gate ablation: does the surprise gate earn its keep, or is it LR scheduling?

The neuromodulatory gate is STC-LoRA's novelty claim, so it must pass the same
discipline we apply everywhere: the ADAPTIVE gate must beat its own MATCHED
NON-ADAPTIVE ablation. Three arms, identical seed/data/config (slow_stc:
stc_full mode + perm_lr_ratio=0.05), differing ONLY in the neuromod signal:

  gated     : m = running-standardized self-loss (the method)
  const-0.5 : m = 0.5, the gate's neutral output -- matched AVERAGE plasticity,
              zero adaptivity. Dream capture still fires (0.5 * SLEEP_BOOST
              1.6 = 0.8 > threshold 0.6), so consolidation survives; only the
              per-chunk adaptivity is removed.
  const-1.0 : m = 1.0 -- maximal plasticity, capture on every chunk ("learn
              hard, consolidate always").

If gated does not beat both on forgetting at comparable final ppl, the gate is
decoration and the paper should say so.

Run:  python -m experiments.gate_ablation
"""

from __future__ import annotations

import json
import time

import numpy as np

import experiments.cl_benchmark as cb
from stc_lora import STCLoRA, STCLoRAConfig
from stc_lora.metrics import compute_cl_summary
from stc_lora.backend import HFCausalLM

PERM_LR = 0.05
SEED = 0


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class ConstGate(cb.StreamGate):
    """A StreamGate whose neuromod is a constant -- the non-adaptive ablation."""

    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def neuromod(self, loss: float) -> float:
        return self.c


def main() -> None:
    log(f"Loading {cb.MODEL_NAME}...")
    llm = HFCausalLM(cb.MODEL_NAME, dtype="float32")
    adapter = STCLoRA(llm.model, STCLoRAConfig(
        rank=8, alpha=16.0, base_lr=cb.BASE_LR, capture_fraction=0.6,
    ))
    domains = cb.load_domains(llm.tokenizer, chunk_len=cb.CHUNK_LEN,
                              train_chunks=cb.TRAIN_CHUNKS,
                              test_chunks=cb.TEST_CHUNKS)
    base = cb.eval_row(llm, domains)
    log("base ppl: " + "  ".join(f"{d.name}={p:.1f}"
                                 for d, p in zip(domains, base)))

    arms = {"gated": None, "const-0.5": 0.5, "const-1.0": 1.0}
    real_gate = cb.StreamGate
    results = {}
    for name, const in arms.items():
        cb.StreamGate = real_gate if const is None else (
            lambda c=const: ConstGate(c))
        adapter.set_perm_plastic(PERM_LR)
        t0 = time.perf_counter()
        r = cb.run_method(llm, adapter, domains, "stc_full", seed=SEED)
        s = compute_cl_summary(r["R"], base=base)
        results[name] = {"forgetting_pct": s["forgetting_pct"],
                         "final_ppl": s["avg_final_ppl"],
                         "learned_ppl": s["avg_learned_ppl"],
                         "captures": r.get("captures")}
        log(f"  {name:9s} forget={s['forgetting_pct']:+6.1f}%  "
            f"final={s['avg_final_ppl']:.2f}  learned={s['avg_learned_ppl']:.2f}"
            f"  captures={r.get('captures')}  [{time.perf_counter()-t0:.0f}s]")
    cb.StreamGate = real_gate
    adapter.set_perm_plastic(0.0)

    g = results["gated"]
    beats = all(g["forgetting_pct"] < results[a]["forgetting_pct"]
                for a in ("const-0.5", "const-1.0"))
    log("")
    log("== GATE ABLATION (slow_stc, identical seed/data; only m differs) ==")
    for n, v in results.items():
        log(f"  {n:9s}: forget {v['forgetting_pct']:+6.1f}%  "
            f"final {v['final_ppl']:.2f}  captures {v['captures']}")
    log(f"VERDICT: gated {'BEATS both ablations -- the gate earns its keep' if beats else 'does NOT beat both -- report honestly / simplify claim'}")

    with open("outputs/gate_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    log("saved outputs/gate_ablation.json")


if __name__ == "__main__":
    main()
