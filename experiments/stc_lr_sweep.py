"""Focused sweep: raise stc_full's learning rate to close the plasticity gap.

Diagnosis (P2): stc_full has the best forgetting (+3.9%) but the WORST
learned_ppl (47.6 vs ER 44.7) — it under-learns each domain because
surprise-gating multiplies base_lr by ~0.5 on average, so it trains at ~half
ER's effective rate. Forgetting has huge headroom (3.9% vs ER's 8.9%), so we
can afford to be more plastic.

This sweeps ONLY stc_full's base_lr (gating still halves it) against a fixed
ER reference bar, at a smaller scale for speed. Target: learned_ppl down near
ER's while forgetting stays below ER's — which would make stc_full win on both.

Run:  python -m experiments.stc_lr_sweep
"""

from __future__ import annotations

import time

import numpy as np

from stc_lora import STCLoRA, STCLoRAConfig
from stc_lora.metrics import compute_cl_summary
from stc_lora.backend import HFCausalLM
import experiments.cl_benchmark as cb

TRAIN_CHUNKS = 150      # smaller than P2's 200 for speed (trend, not final)
TEST_CHUNKS = 50
LR_GRID = [0.05, 0.10, 0.15]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"Loading {cb.MODEL_NAME}...")
    llm = HFCausalLM(cb.MODEL_NAME, dtype="float32")
    adapter = STCLoRA(llm.model, STCLoRAConfig(
        rank=8, alpha=16.0, base_lr=cb.BASE_LR, capture_fraction=0.6,
    ))
    domains = cb.load_domains(llm.tokenizer, chunk_len=cb.CHUNK_LEN,
                              train_chunks=TRAIN_CHUNKS, test_chunks=TEST_CHUNKS)
    base = cb.eval_row(llm, domains)
    log(f"base ppl: " + "  ".join(f"{n}={p:.1f}" for n, p in
                                  zip(cb.DOMAIN_ORDER, base)))

    # Reference bar: ER (plain replay), fixed at BASE_LR.
    log("== ER reference ==")
    r = cb.run_method(llm, adapter, domains, "er")
    ser = compute_cl_summary(r["R"], base=base)
    log(f"  ER: learned={ser['avg_learned_ppl']:.2f}  "
        f"forget={ser['forgetting_pct']:+.1f}%  final={ser['avg_final_ppl']:.2f}")

    rows = []
    for lr in LR_GRID:
        adapter.config.base_lr = lr        # stc uses config.base_lr (x gating)
        r = cb.run_method(llm, adapter, domains, "stc_full")
        s = compute_cl_summary(r["R"], base=base)
        rows.append((lr, s, r.get("captures", 0)))
        log(f"  stc_full lr={lr:.2f}: learned={s['avg_learned_ppl']:.2f}  "
            f"forget={s['forgetting_pct']:+.1f}%  final={s['avg_final_ppl']:.2f}  "
            f"captures={r.get('captures', 0)}")

    log("== Summary (vs ER bar: learned=%.1f forget=%+.1f%% final=%.1f) ==" % (
        ser["avg_learned_ppl"], ser["forgetting_pct"], ser["avg_final_ppl"]))
    for lr, s, caps in rows:
        beats_forget = s["forgetting_pct"] < ser["forgetting_pct"]
        beats_final = s["avg_final_ppl"] < ser["avg_final_ppl"]
        tag = ("BEATS ER on BOTH" if beats_forget and beats_final
               else "beats forget only" if beats_forget
               else "beats final only" if beats_final else "-")
        log(f"  lr={lr:.2f}  learned={s['avg_learned_ppl']:.1f}  "
            f"forget={s['forgetting_pct']:+.1f}%  final={s['avg_final_ppl']:.1f}   {tag}")


if __name__ == "__main__":
    main()
