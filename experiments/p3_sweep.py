"""P3: multi-seed x task-order sweep with error bars.

Turns the single-run P1/P2 numbers into mean +/- std by replicating each method
across several (seed, task-order) draws. Continual learning is order-sensitive,
so each replicate uses a DIFFERENT domain order; averaging backward-transfer
over orders gives an order-robust forgetting estimate, and final-avg-perplexity
is order-invariant by construction.

Methods (all rank-8, one shared adapter toggled per run):
  naive, ewc, er, stc_frozen (perm frozen), slow_stc (perm_lr_ratio=0.05).
Plus joint once as the order-invariant ceiling.

Model via MODEL env var (default Qwen); run once per model, then aggregate.
Results are written incrementally so a partial run is still usable.

Run:  python -m experiments.p3_sweep            # Qwen2.5-0.5B
      MODEL=meta-llama/Llama-3.2-1B python -m experiments.p3_sweep
"""

from __future__ import annotations

import json
import os
import statistics as stats
import time
from pathlib import Path

from stc_lora import STCLoRA, STCLoRAConfig
from stc_lora.metrics import compute_cl_summary
from stc_lora.backend import HFCausalLM
import experiments.cl_benchmark as cb

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
DTYPE = os.environ.get("DTYPE", "float32")   # bf16 for larger models on 8GB
TRAIN_CHUNKS = 200
TEST_CHUNKS = 50
PERM_LR = 0.05

# (seed, order) replicates — orders are permutations of the 3 domains so the
# forgetting estimate is not tied to one arrangement.
ORDERS = {
    "A": ["news", "wiki", "shake"],
    "B": ["shake", "news", "wiki"],
    "C": ["wiki", "shake", "news"],
    "D": ["news", "shake", "wiki"],
}
REPLICATES = [(0, "A"), (1, "B"), (2, "C"), (3, "D")]
METHODS = ["naive", "ewc", "er", "stc_frozen", "slow_stc"]

OUT_PATH = Path(f"outputs/p3_{MODEL.split('/')[-1]}.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"Loading {MODEL} (dtype={DTYPE})...")
    llm = HFCausalLM(MODEL, dtype=DTYPE)
    adapter = STCLoRA(llm.model, STCLoRAConfig(
        rank=8, alpha=16.0, base_lr=cb.BASE_LR, capture_fraction=0.6,
    ))
    log(f"adapter on {len(adapter.layers)} modules, {llm.device}")

    by_name = {d.name: d for d in cb.load_domains(
        llm.tokenizer, chunk_len=cb.CHUNK_LEN,
        train_chunks=TRAIN_CHUNKS, test_chunks=TEST_CHUNKS)}

    runs = []
    OUT_PATH.parent.mkdir(exist_ok=True)

    def do(name, mode, perm, seed, order_key):
        domains = [by_name[n] for n in ORDERS[order_key]]
        base = cb.eval_row(llm, domains)
        adapter.set_perm_plastic(perm)
        t0 = time.perf_counter()
        r = cb.run_method(llm, adapter, domains, mode, seed=seed)
        s = compute_cl_summary(r["R"], base=base)
        rec = {
            "method": name, "seed": seed, "order": order_key,
            "forgetting_pct": s["forgetting_pct"],
            "final_ppl": s["avg_final_ppl"],
            "learned_ppl": s["avg_learned_ppl"],
            "backward_transfer": s["backward_transfer"],
        }
        runs.append(rec)
        OUT_PATH.write_text(json.dumps(runs, indent=2), encoding="utf-8")
        log(f"  {name:11s} seed={seed} order={order_key}: "
            f"forget={s['forgetting_pct']:+6.1f}%  final={s['avg_final_ppl']:.2f}"
            f"  [{time.perf_counter()-t0:.0f}s]")

    # Ceiling: joint is order-invariant, run once (order A, seed 0).
    log("== joint ceiling (once) ==")
    do("joint", "joint", 0.0, 0, "A")

    for seed, order_key in REPLICATES:
        log(f"== replicate seed={seed} order={order_key} ==")
        for m in METHODS:
            if m == "slow_stc":
                do("slow_stc", "stc_full", PERM_LR, seed, order_key)
            elif m == "stc_frozen":
                do("stc_frozen", "stc_full", 0.0, seed, order_key)
            else:
                do(m, m, 0.0, seed, order_key)
    adapter.set_perm_plastic(0.0)

    # -- aggregate mean +/- std per method --
    log(f"== {MODEL} aggregate (mean +/- std over {len(REPLICATES)} replicates) ==")
    agg = {}
    for m in ["naive", "ewc", "er", "stc_frozen", "slow_stc"]:
        fs = [r["forgetting_pct"] for r in runs if r["method"] == m]
        ps = [r["final_ppl"] for r in runs if r["method"] == m]
        if not fs:
            continue
        fmean, fsd = stats.mean(fs), (stats.stdev(fs) if len(fs) > 1 else 0.0)
        pmean, psd = stats.mean(ps), (stats.stdev(ps) if len(ps) > 1 else 0.0)
        agg[m] = {"forget_mean": fmean, "forget_std": fsd,
                  "final_mean": pmean, "final_std": psd, "n": len(fs)}
        log(f"  {m:11s} forget={fmean:+5.1f} +/- {fsd:4.1f}%   "
            f"final_ppl={pmean:.1f} +/- {psd:.1f}")
    joint = next((r["final_ppl"] for r in runs if r["method"] == "joint"), None)
    if joint is not None:
        log(f"  joint ceiling final_ppl={joint:.1f}")

    OUT_PATH.write_text(json.dumps(
        {"model": MODEL, "runs": runs, "aggregate": agg,
         "joint_ceiling": joint}, indent=2), encoding="utf-8")
    log(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
