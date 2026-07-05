"""Focused test: does a SLOW-PLASTIC permanent store close stc_full's ppl gap?

P2 diagnosis + LR sweep showed stc_full under-learns (learned_ppl worse than
ER) NOT because of learning rate, but because the permanent store is FROZEN —
knowledge captured mid-domain is locked at immature quality. This tries the
real fix: let perm_delta keep refining by gradient at a slow rate
(fast/slow-weights), so consolidated knowledge improves instead of petrifying.

Compares, on the same LoRA substrate at a small scale for speed:
  - ER              (reference bar)
  - stc_full frozen (perm_lr_ratio = 0, original)
  - stc_full slow   (perm_lr_ratio in a small grid)

Target: slow-plastic drops learned_ppl toward ER's while keeping forgetting
below ER's -> beats ER on both.

Run:  python -m experiments.perm_plastic_sweep
"""

from __future__ import annotations

import time

from stc_lora import STCLoRA, STCLoRAConfig
from stc_lora.metrics import compute_cl_summary
from stc_lora.backend import HFCausalLM
import experiments.cl_benchmark as cb

TRAIN_CHUNKS = 120
TEST_CHUNKS = 50
PERM_LR_GRID = [0.1, 0.3]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def summarize(name, r, base, ser=None):
    s = compute_cl_summary(r["R"], base=base)
    tag = ""
    if ser is not None:
        bf = s["forgetting_pct"] < ser["forgetting_pct"]
        bl = s["avg_final_ppl"] < ser["avg_final_ppl"]
        tag = ("  BEATS ER on BOTH" if bf and bl else
               "  beats forget only" if bf else
               "  beats final only" if bl else "")
    log(f"  {name:18s} learned={s['avg_learned_ppl']:.2f}  "
        f"forget={s['forgetting_pct']:+.1f}%  final={s['avg_final_ppl']:.2f}"
        f"{tag}")
    return s


def main() -> None:
    log(f"Loading {cb.MODEL_NAME}...")
    llm = HFCausalLM(cb.MODEL_NAME, dtype="float32")
    adapter = STCLoRA(llm.model, STCLoRAConfig(
        rank=8, alpha=16.0, base_lr=cb.BASE_LR, capture_fraction=0.6,
    ))
    domains = cb.load_domains(llm.tokenizer, chunk_len=cb.CHUNK_LEN,
                              train_chunks=TRAIN_CHUNKS, test_chunks=TEST_CHUNKS)
    base = cb.eval_row(llm, domains)
    log("base: " + "  ".join(f"{n}={p:.1f}" for n, p in
                             zip(cb.DOMAIN_ORDER, base)))

    log("== ER reference ==")
    adapter.set_perm_plastic(0.0)
    ser = summarize("er", cb.run_method(llm, adapter, domains, "er"), base)

    log("== stc_full frozen permanent (perm_lr_ratio=0) ==")
    adapter.set_perm_plastic(0.0)
    summarize("stc_full frozen", cb.run_method(llm, adapter, domains, "stc_full"),
              base, ser)

    log("== stc_full SLOW-PLASTIC permanent ==")
    for ratio in PERM_LR_GRID:
        adapter.set_perm_plastic(ratio)
        summarize(f"stc_slow r={ratio}",
                  cb.run_method(llm, adapter, domains, "stc_full"), base, ser)
    adapter.set_perm_plastic(0.0)


if __name__ == "__main__":
    main()
