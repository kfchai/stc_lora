"""A-GEM baseline (Chaudhry et al., 2019) on the shared adapter substrate.

Averaged Gradient Episodic Memory: before each step, compute the gradient on
the current chunk and a REFERENCE gradient on a small batch from episodic
memory; if they conflict (negative dot product), project the current gradient
onto the half-space that does not increase memory loss:

    g <- g - (g . g_ref / ||g_ref||^2) g_ref     when  g . g_ref < 0

Same LoRA substrate, same BASE_LR, and the SAME Vitter reservoir as ER/dream --
so the only difference is the algorithm. Compute is ~2x naive (one extra
backward per step), comparable to ER's matched replay budget.

Run:  python -m experiments.agem_baseline        # single run, seed 0, order A
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

import experiments.cl_benchmark as cb
from stc_lora import STCLoRA, STCLoRAConfig
from stc_lora.metrics import compute_cl_summary
from stc_lora.backend import HFCausalLM

AGEM_REF_BATCH = 4
SEED = 0


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run_agem(llm, adapter, domains, rng) -> dict:
    adapter.config.decay_rate = 0.0
    adapter.config.capture_threshold = 1e9
    adapter.reset_all()
    reservoir = cb.ReservoirSampler(cb.RESERVOIR_CAP, rng)
    params = list(adapter.trainable_parameters())
    T = len(domains)
    R = np.zeros((T, T))
    n_proj = 0

    for i, dom in enumerate(domains):
        for ids in dom.train_chunks:
            loss = adapter.lm_loss(torch.tensor([ids], device=llm.device))
            adapter._opt.zero_grad(set_to_none=True)
            loss.backward()
            g = [p.grad.detach().clone() for p in params]

            if reservoir.items:
                k = min(AGEM_REF_BATCH, len(reservoir.items))
                pick = rng.choice(len(reservoir.items), size=k, replace=False)
                mem = torch.tensor([reservoir.items[j] for j in pick],
                                   device=llm.device)
                mloss = adapter.lm_loss(mem)
                adapter._opt.zero_grad(set_to_none=True)
                mloss.backward()
                gref = [p.grad.detach().clone() for p in params]
                dot = sum((a * b).sum() for a, b in zip(g, gref))
                if dot < 0:
                    ref2 = sum((b * b).sum() for b in gref) + 1e-12
                    coef = dot / ref2
                    g = [a - coef * b for a, b in zip(g, gref)]
                    n_proj += 1

            with torch.no_grad():
                for p, gr in zip(params, g):
                    p.grad = gr
                torch.nn.utils.clip_grad_norm_(params, adapter.config.grad_clip)
            for grp in adapter._opt.param_groups:
                grp["lr"] = cb.BASE_LR
            adapter._opt.step()
            llm.model.eval()
            reservoir.offer(ids)
        R[i] = cb.eval_row(llm, domains)

    return {"R": R.tolist(), "projections": n_proj}


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
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    r = run_agem(llm, adapter, domains, rng)
    s = compute_cl_summary(r["R"], base=base)
    log(f"A-GEM: forget={s['forgetting_pct']:+.1f}%  "
        f"final={s['avg_final_ppl']:.2f}  learned={s['avg_learned_ppl']:.2f}  "
        f"projections={r['projections']}  [{time.perf_counter()-t0:.0f}s]")
    log("(0.5B seed-0 comparators: naive +39.2, ewc +21.3, er +11.4, "
        "stc_frozen +3.8, slow_stc +5.4 — P3 means)")
    with open("outputs/agem_baseline.json", "w") as f:
        json.dump({"summary": s, "projections": r["projections"]}, f, indent=2)
    log("saved outputs/agem_baseline.json")


if __name__ == "__main__":
    main()
