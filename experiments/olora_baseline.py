"""O-LoRA baseline (Wang et al., EMNLP 2023) on the shared adapter substrate.

Orthogonal subspace learning: each task trains a FRESH LoRA pair whose input
subspace (rows of A) is penalized toward orthogonality with all PREVIOUS
tasks' A matrices; finished tasks' deltas are frozen and summed at inference.
Interference is prevented SPATIALLY (orthogonal directions), where STC-LoRA
prevents it TEMPORALLY (decay vs consolidation).

Faithful adaptation to our benchmark, with two deliberate concessions TO
O-LoRA (both advantages over our method's conditions, stated in the paper):
  1. ORACLE task boundaries -- it is told exactly when each domain ends
     (it cannot run boundary-free; our method gets no boundaries).
  2. Growing capacity -- a fresh rank-8 pair per domain (3x the tagged
     capacity our method uses), which is how O-LoRA is designed.

Mechanics on the substrate: at each oracle boundary, the current tagged delta
is consolidated verbatim into perm_delta (= frozen sum of past tasks) and the
tagged pair is re-randomized (a fresh subspace). During task t, loss =
LM loss + lambda * sum_layers sum_{i<t} ||A_i A_t^T||_F^2.

Run:  python -m experiments.olora_baseline           # sweeps lambda in {0.1, 0.5, 2.0}
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

LAMBDAS = [0.1, 0.5, 2.0]
SEED = 0


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run_olora(llm, adapter, domains, lam: float, seed: int = 0) -> dict:
    adapter.config.decay_rate = 0.0            # no STC dynamics
    adapter.config.capture_threshold = 1e9
    adapter.config.capture_fraction = 1.0      # boundary freeze moves ALL mass
    adapter.reset_all()
    gen = torch.Generator(device="cpu").manual_seed(seed)

    frozen_As: list[list[torch.Tensor]] = []   # per past task: per layer A
    T = len(domains)
    R = np.zeros((T, T))

    for i, dom in enumerate(domains):
        for ids in dom.train_chunks:
            t = torch.tensor([ids], device=llm.device)
            loss = adapter.lm_loss(t)
            if frozen_As:                      # orthogonality to past subspaces
                pen = 0.0
                for l_idx, layer in enumerate(adapter.layers):
                    A_t = layer.A_tag
                    for past in frozen_As:
                        pen = pen + (past[l_idx] @ A_t.T).pow(2).sum()
                loss = loss + lam * pen
            adapter.sgd_step(loss, cb.BASE_LR)
        # ORACLE boundary: freeze this task's delta into the summed store,
        # remember its A subspace, start a fresh random pair for the next task.
        frozen_As.append([l.A_tag.detach().clone() for l in adapter.layers])
        with torch.no_grad():
            for l in adapter.layers:
                l.capture(1.0)                 # perm += s*(B@A); B -> 0
                l.A_tag.copy_(torch.randn(
                    l.A_tag.shape, generator=gen).to(l.A_tag) * 0.01)
        R[i] = cb.eval_row(llm, domains)

    # sanity: how orthogonal did consecutive subspaces end up?
    with torch.no_grad():
        cross = float(np.mean([
            (frozen_As[a][l] @ frozen_As[b][l].T).pow(2).sum().item()
            for l in range(len(adapter.layers))
            for a in range(len(frozen_As)) for b in range(a + 1, len(frozen_As))
        ])) if len(frozen_As) > 1 else 0.0
    return {"R": R.tolist(), "cross_subspace_energy": cross}


def main() -> None:
    log(f"Loading {cb.MODEL_NAME}...")
    llm = HFCausalLM(cb.MODEL_NAME, dtype="float32")
    adapter = STCLoRA(llm.model, STCLoRAConfig(
        rank=8, alpha=16.0, base_lr=cb.BASE_LR, capture_fraction=1.0,
    ))
    domains = cb.load_domains(llm.tokenizer, chunk_len=cb.CHUNK_LEN,
                              train_chunks=cb.TRAIN_CHUNKS,
                              test_chunks=cb.TEST_CHUNKS)
    base = cb.eval_row(llm, domains)

    results = {}
    for lam in LAMBDAS:
        t0 = time.perf_counter()
        r = run_olora(llm, adapter, domains, lam, seed=SEED)
        s = compute_cl_summary(r["R"], base=base)
        results[f"lambda_{lam}"] = {"summary": s,
                                    "cross_energy": r["cross_subspace_energy"]}
        log(f"  O-LoRA lam={lam:<4} forget={s['forgetting_pct']:+6.1f}%  "
            f"final={s['avg_final_ppl']:.2f}  learned={s['avg_learned_ppl']:.2f}"
            f"  cross={r['cross_subspace_energy']:.3f}"
            f"  [{time.perf_counter()-t0:.0f}s]")

    log("(0.5B comparators, P3 means: naive +39.2/54.7, ewc +21.3/50.0, "
        "er +11.4/47.8, agem +23.4/49.9, stc_frozen +3.8/49.2, "
        "slow_stc +5.4/47.5)")
    with open("outputs/olora_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    log("saved outputs/olora_baseline.json")


if __name__ == "__main__":
    main()
