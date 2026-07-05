"""Capacity-fair check: is slow-plastic's ppl parity mechanism or params?

slow-stc (rank-8 tag + dense plastic permanent, perm_lr_ratio=0.05) trains
~22.5M params; ER rank-8 trains 540K. So slow-stc's ppl parity with ER could
be "40x more trainable weights" rather than the fast/slow mechanism.

This gives ER MORE capacity — LoRA ranks up to the one that matches slow-stc's
22.5M trainable params (rank ~333) — and asks:
  - Does ER's forgetting fall as its capacity rises? (If yes, capacity explains
    the ppl gap and slow-stc has no real edge.)
  - Or does slow-stc still forget far less at EQUAL parameter count? (Then the
    forgetting resistance is the mechanism, not capacity.)

Standard result: more trainable capacity usually makes forgetting WORSE, so the
prediction is ER stays high-forgetting while slow-stc (same params) stays low.

Full 200-chunk scale. Fresh model per config (injection is in-place).

Run:  python -m experiments.capacity_fair
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from stc_lora import STCLoRA, STCLoRAConfig
from stc_lora.metrics import compute_cl_summary
from stc_lora.backend import HFCausalLM
import experiments.cl_benchmark as cb

TRAIN_CHUNKS = 200
TEST_CHUNKS = 50
ER_RANKS = [8, 64, 333]      # 333 -> ~22.5M params, matched to slow-stc
OUT_PATH = Path("outputs/capacity_fair.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(rank: int, perm_lr_ratio: float):
    llm = HFCausalLM(cb.MODEL_NAME, dtype="float32")
    adapter = STCLoRA(llm.model, STCLoRAConfig(
        rank=rank, alpha=2.0 * rank, base_lr=cb.BASE_LR,
        capture_fraction=0.6, perm_lr_ratio=perm_lr_ratio,
    ))
    tag = sum(p.numel() for p in adapter.trainable_parameters())
    perm = (sum(l.perm_delta.numel() for l in adapter.layers)
            if perm_lr_ratio > 0 else 0)
    return llm, adapter, tag + perm


def main() -> None:
    # Load domains once from an initial tokenizer (identical across reloads).
    log("Loading domains...")
    llm0 = HFCausalLM(cb.MODEL_NAME, dtype="float32")
    domains = cb.load_domains(llm0.tokenizer, chunk_len=cb.CHUNK_LEN,
                              train_chunks=TRAIN_CHUNKS, test_chunks=TEST_CHUNKS)
    del llm0

    results = {}
    configs = [(f"er_rank{r}", "er", r, 0.0) for r in ER_RANKS]
    configs.append(("slow_stc_r0.05", "stc_full", 8, 0.05))

    for name, mode, rank, perm in configs:
        llm, adapter, n_params = build(rank, perm)
        base = cb.eval_row(llm, domains)
        t0 = time.perf_counter()
        r = cb.run_method(llm, adapter, domains, mode)
        s = compute_cl_summary(r["R"], base=base)
        results[name] = {"summary": s, "n_trainable": n_params}
        log(f"  {name:16s} params={n_params/1e6:5.2f}M  "
            f"learned={s['avg_learned_ppl']:.2f}  "
            f"forget={s['forgetting_pct']:+6.1f}%  "
            f"final={s['avg_final_ppl']:.2f}  [{time.perf_counter()-t0:.0f}s]")
        del llm, adapter

    log("== Verdict: does ER's forgetting fall with capacity? ==")
    for name, d in results.items():
        s = d["summary"]
        log(f"  {name:16s} {d['n_trainable']/1e6:6.2f}M  "
            f"forget={s['forgetting_pct']:+6.1f}%  final={s['avg_final_ppl']:.1f}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
