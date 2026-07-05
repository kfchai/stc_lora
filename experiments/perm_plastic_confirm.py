"""Confirmatory run: slow-plastic permanent at FULL interference scale.

The 120-chunk sweep showed slow-plastic fixes stc_full's learning deficit but
trades forgetting for it, and at that small scale ER's forgetting was already
near-best so nothing dominated. BUT at P2's 200-chunk scale frozen stc_full
forgot FAR less than ER (3.9% vs 8.9%) — a large stability headroom. This tests
whether spending a LITTLE of that headroom (small perm_lr_ratio) recovers the
perplexity gap while keeping forgetting under ER — i.e. beats ER on BOTH.

Bars: ER (rehearsal baseline) and frozen stc_full (max stability).
Grid: perm_lr_ratio in {0.05, 0.10}.

Run:  python -m experiments.perm_plastic_confirm
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from stc_lora import STCLoRA, STCLoRAConfig
from stc_lora.metrics import compute_cl_summary
from stc_lora.backend import HFCausalLM
import experiments.cl_benchmark as cb

TRAIN_CHUNKS = 200          # full P2 scale
TEST_CHUNKS = 50
PERM_LR_GRID = [0.05, 0.10]
OUT_PATH = Path("outputs/perm_plastic_confirm.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"Loading {cb.MODEL_NAME}...")
    llm = HFCausalLM(cb.MODEL_NAME, dtype="float32")
    adapter = STCLoRA(llm.model, STCLoRAConfig(
        rank=8, alpha=16.0, base_lr=cb.BASE_LR, capture_fraction=0.6,
    ))
    tag_params = sum(p.numel() for p in adapter.trainable_parameters())
    perm_params = sum(l.perm_delta.numel() for l in adapter.layers)
    log(f"params: tag(low-rank)={tag_params:,}  perm(dense)={perm_params:,}")
    log("  ER/frozen train the tag only; slow-plastic also refines the dense "
        "perm (already present as a buffer, now unfrozen).")

    domains = cb.load_domains(llm.tokenizer, chunk_len=cb.CHUNK_LEN,
                              train_chunks=TRAIN_CHUNKS, test_chunks=TEST_CHUNKS)
    base = cb.eval_row(llm, domains)
    log("base: " + "  ".join(f"{n}={p:.1f}" for n, p in
                             zip(cb.DOMAIN_ORDER, base)))

    results = {}

    def run(name, mode, perm_ratio):
        adapter.set_perm_plastic(perm_ratio)
        t0 = time.perf_counter()
        r = cb.run_method(llm, adapter, domains, mode)
        s = compute_cl_summary(r["R"], base=base)
        results[name] = {"summary": s, "captures": r.get("captures", 0)}
        log(f"  {name:16s} learned={s['avg_learned_ppl']:.2f}  "
            f"forget={s['forgetting_pct']:+.1f}%  final={s['avg_final_ppl']:.2f}"
            f"  [{time.perf_counter() - t0:.0f}s]")
        return s

    log("== bars ==")
    ser = run("er", "er", 0.0)
    sfr = run("stc_frozen", "stc_full", 0.0)

    log("== slow-plastic permanent ==")
    for ratio in PERM_LR_GRID:
        run(f"stc_slow_{ratio}", "stc_full", ratio)
    adapter.set_perm_plastic(0.0)

    log("== Verdict (vs ER: forget=%.1f%% final=%.1f) ==" % (
        ser["forgetting_pct"], ser["avg_final_ppl"]))
    for name, d in results.items():
        s = d["summary"]
        bf = s["forgetting_pct"] < ser["forgetting_pct"]
        bl = s["avg_final_ppl"] < ser["avg_final_ppl"]
        tag = ("BEATS ER on BOTH" if bf and bl else
               "beats forgetting" if bf else
               "beats final ppl" if bl else "-")
        log(f"  {name:16s} forget={s['forgetting_pct']:+6.1f}%  "
            f"final={s['avg_final_ppl']:.1f}   {tag}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
