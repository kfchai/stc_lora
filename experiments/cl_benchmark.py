"""Track A #3 / P1: domain-incremental continual-learning benchmark.

Online, boundary-free streaming over three distinct registers
(news -> encyclopedia -> Elizabethan verse). The learner sees one continuous
token stream and is never told where a domain ends; evaluation snapshots
perplexity on every domain after each domain is consumed, building the
task-by-task R-matrix used for backward/forward transfer.

P1 methods (all share ONE LoRA substrate — same rank/targets/optimizer):
  - naive : plain sequential LoRA (no decay/capture/gating/dream). Lower bound.
  - joint : all domains shuffled together, single pass. Upper bound (no order).
  - stc_nodream : surprise-gated plasticity + tag decay + capture. No replay.
  - stc_full    : the above + scheduled dream replay of a high-surprise
                  reservoir (boundary-free: fires every DREAM_EVERY steps).

Plasticity is gated on the model's OWN predictive surprise (per-chunk LM loss,
running-normalized) with a one-step delay — no separate embedder in the loop.

Run:  python -m experiments.cl_benchmark
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from stc_lora import STCLoRA, STCLoRAConfig
from stc_lora.metrics import compute_cl_summary
from stc_lora.backend import HFCausalLM
from experiments.cl_data import DOMAIN_ORDER, load_domains

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

CHUNK_LEN = 128
TRAIN_CHUNKS = 200
TEST_CHUNKS = 50
BASE_LR = 5e-2
DREAM_EVERY = 100            # scheduled dream cadence (boundary-free)
DREAM_ROUNDS = 2
RESERVOIR_CAP = 40
SLEEP_BOOST = 1.6

# Baselines. ER's replay budget is matched to stc_full's dream budget:
# stc_full replays ~ (600/DREAM_EVERY) * DREAM_ROUNDS * RESERVOIR_CAP steps.
# With 3x200 chunks that is 6*2*40 = 480 replays over 600 new steps -> 0.8.
ER_REPLAY_RATIO = 0.8        # old-example replay steps per new step
EWC_LAMBDA = 1e2             # EWC penalty strength (tuned: learns ~like naive
                             # while cutting forgetting; higher -> underfits)
EWC_GAMMA = 0.95            # online-Fisher decay
EWC_ANCHOR_EVERY = 100       # steps between anchor refreshes (boundary-free)
OUT_PATH = Path("outputs/cl_benchmark_results.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ReservoirSampler:
    """Vitter Algorithm R: a uniform, position-unbiased sample of a stream.

    Every item offered so far has equal probability cap/n of being retained,
    so the buffer stays BALANCED across the whole stream — all domains equally
    represented — instead of skewing toward recent items. This is the P1 fix:
    the previous recency-skewed reservoir made dream replay rehearse mostly the
    latest domain, which accelerated forgetting of earlier ones.
    """

    def __init__(self, cap: int, rng):
        self.cap = cap
        self.rng = rng
        self.items: list = []
        self.seen = 0

    def offer(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.cap:
            self.items.append(item)
        else:
            j = int(self.rng.integers(self.seen))
            if j < self.cap:
                self.items[j] = item


class StreamGate:
    """Maps the model's per-chunk LM loss to a neuromod signal in [0, 1].

    Running mean/std of loss; a chunk whose loss is above average is
    'surprising' and earns more plasticity. Average -> 0.5, boring -> low,
    surprising -> high. Used with a one-step delay.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0
        self.n = 0

    def update(self, loss: float) -> None:
        self.n += 1
        if self.n == 1:
            self.mean = loss
            return
        d = loss - self.mean
        self.mean += self.alpha * d
        self.var = (1 - self.alpha) * (self.var + self.alpha * d * d)

    def neuromod(self, loss: float) -> float:
        std = max(math.sqrt(self.var), 1e-6)
        z = (loss - self.mean) / std
        return float(np.clip(0.5 + 0.5 * np.tanh(z), 0.05, 1.0))


@torch.no_grad()
def chunk_nll(llm, ids_list) -> float:
    """Mean per-token NLL over a list of token-id chunks."""
    device = llm.device
    total = 0.0
    for ids in ids_list:
        t = torch.tensor([ids], device=device)
        out = llm.model(t, labels=t)
        total += float(out.loss)
    return total / max(len(ids_list), 1)


def ppl(llm, ids_list) -> float:
    return float(math.exp(min(chunk_nll(llm, ids_list), 20.0)))


def eval_row(llm, domains) -> list[float]:
    """Perplexity on every domain's held-out test set (one R-matrix row)."""
    return [ppl(llm, d.test_chunks) for d in domains]


def dream_replay(llm, adapter, reservoir, rounds, boost, rng):
    """Replay reservoir chunks with sleep-boosted neuromod to drive capture."""
    if not reservoir:
        return
    for _ in range(rounds):
        order = rng.permutation(len(reservoir))
        for idx in order:
            ids, nm = reservoir[idx]
            t = torch.tensor([ids], device=llm.device)
            adapter.learn(t, neuromod=min(1.0, nm * boost))


def run_er(llm, adapter, domains, rng) -> dict:
    """Experience Replay baseline: plain LoRA + balanced-reservoir rehearsal.

    Same LoRA substrate as everyone else, no STC dynamics. Each new chunk gets
    a normal SGD step; a running 'replay debt' (ER_REPLAY_RATIO per step)
    triggers extra SGD steps on random buffered chunks. The buffer is the SAME
    Vitter reservoir our dream uses — so ER and dream face an equal, equally
    balanced replay budget, and the only difference is HOW they replay
    (dream consolidates into permanent weights; ER just re-trains the tag).
    """
    adapter.config.decay_rate = 0.0
    adapter.config.capture_threshold = 1e9
    adapter.reset_all()
    reservoir = ReservoirSampler(RESERVOIR_CAP, rng)
    debt = 0.0
    T = len(domains)
    R = np.zeros((T, T))
    n_replays = 0
    for i, dom in enumerate(domains):
        for ids in dom.train_chunks:
            adapter.sgd_step(adapter.lm_loss(
                torch.tensor([ids], device=llm.device)), BASE_LR)
            reservoir.offer(ids)
            debt += ER_REPLAY_RATIO
            while debt >= 1.0 and reservoir.items:
                old = reservoir.items[int(rng.integers(len(reservoir.items)))]
                adapter.sgd_step(adapter.lm_loss(
                    torch.tensor([old], device=llm.device)), BASE_LR)
                debt -= 1.0
                n_replays += 1
        R[i] = eval_row(llm, domains)
    return {"R": R.tolist(), "replays": n_replays}


def run_ewc(llm, adapter, domains, rng) -> dict:
    """Online-EWC baseline: plain LoRA + a quadratic penalty anchoring the
    weights that matter most for what was already learned.

    Boundary-free (Schwarz et al. 2018): a running diagonal Fisher (decayed by
    EWC_GAMMA) estimates per-weight importance from the LM gradient; an anchor
    of consolidated weights is refreshed every EWC_ANCHOR_EVERY steps. The loss
    is LM loss + (lambda/2) * sum F_i (theta_i - anchor_i)^2, so changing an
    important weight is expensive. No stored data (its whole selling point).
    """
    adapter.config.decay_rate = 0.0
    adapter.config.capture_threshold = 1e9
    adapter.reset_all()

    params = list(adapter.trainable_parameters())
    fisher = [torch.zeros_like(p) for p in params]
    anchor = [p.detach().clone() for p in params]
    T = len(domains)
    R = np.zeros((T, T))
    step = 0
    for i, dom in enumerate(domains):
        for ids in dom.train_chunks:
            t = torch.tensor([ids], device=llm.device)
            adapter._opt.zero_grad(set_to_none=True)
            lm = adapter.lm_loss(t)
            lm.backward(retain_graph=True)
            # snapshot LM-only gradients for the Fisher estimate
            lm_g2 = [(p.grad.detach() ** 2 if p.grad is not None
                      else torch.zeros_like(p)) for p in params]
            # add the EWC penalty gradient on top of the LM gradient
            penalty = 0.0
            for p, f, a in zip(params, fisher, anchor):
                penalty = penalty + (f * (p - a) ** 2).sum()
            (0.5 * EWC_LAMBDA * penalty).backward()
            torch.nn.utils.clip_grad_norm_(params, adapter.config.grad_clip)
            for g in adapter._opt.param_groups:
                g["lr"] = BASE_LR
            adapter._opt.step()
            adapter.model.eval()
            # online Fisher + periodic anchor refresh
            for k in range(len(params)):
                fisher[k] = EWC_GAMMA * fisher[k] + lm_g2[k]
            step += 1
            if step % EWC_ANCHOR_EVERY == 0:
                anchor = [p.detach().clone() for p in params]
        R[i] = eval_row(llm, domains)
    return {"R": R.tolist()}


def run_method(llm, adapter, domains, mode: str, seed: int = 0) -> dict:
    rng0 = np.random.default_rng(seed)
    if mode == "er":
        return run_er(llm, adapter, domains, rng0)
    if mode == "ewc":
        return run_ewc(llm, adapter, domains, rng0)

    stc = mode.startswith("stc")
    do_dream = mode == "stc_full"
    if stc:
        adapter.config.decay_rate = 0.08
        adapter.config.capture_threshold = 0.6
    else:
        adapter.config.decay_rate = 0.0
        adapter.config.capture_threshold = 1e9
    adapter.reset_all()

    rng = np.random.default_rng(seed)
    gate = StreamGate()
    reservoir = ReservoirSampler(RESERVOIR_CAP, rng)
    prev_loss = None
    step = 0
    T = len(domains)
    R = np.zeros((T, T))

    if mode == "joint":
        # Upper bound: all domains shuffled together, single pass.
        allchunks = [c for d in domains for c in d.train_chunks]
        rng.shuffle(allchunks)
        for ids in allchunks:
            t = torch.tensor([ids], device=llm.device)
            adapter.learn(t, neuromod=1.0)
        row = eval_row(llm, domains)
        # Fill every row identically — joint has no task order, so BWT == 0.
        for i in range(T):
            R[i] = row
        return {"R": R.tolist(), "avg_final_ppl": float(np.mean(row))}

    # Sequential streaming (naive / stc_*).
    for i, dom in enumerate(domains):
        for ids in dom.train_chunks:
            t = torch.tensor([ids], device=llm.device)
            if stc:
                nm = gate.neuromod(prev_loss) if prev_loss is not None else 0.5
            else:
                nm = 1.0
            info = adapter.learn(t, neuromod=nm)
            prev_loss = info["loss"]
            if stc:
                gate.update(info["loss"])
                # Offer EVERY chunk to the reservoir (temporally uniform); the
                # stored neuromod still weights how strongly it drives capture
                # during replay, so surprise matters for consolidation strength
                # without biasing which chunks are kept.
                reservoir.offer((ids, nm))
            step += 1
            if do_dream and step % DREAM_EVERY == 0:
                dream_replay(llm, adapter, reservoir.items, DREAM_ROUNDS,
                             SLEEP_BOOST, rng)
        R[i] = eval_row(llm, domains)   # snapshot after finishing domain i

    return {"R": R.tolist(), "captures": adapter.n_captures,
            "perm_norm": adapter.perm_norm(), "reservoir": len(reservoir.items)}


def main() -> None:
    log(f"Loading {MODEL_NAME}...")
    llm = HFCausalLM(MODEL_NAME, dtype="float32")
    adapter = STCLoRA(llm.model, STCLoRAConfig(
        rank=8, alpha=16.0, base_lr=BASE_LR, capture_fraction=0.6,
    ))
    log(f"STC-LoRA on {len(adapter.layers)} modules, {llm.device}")

    log("Loading 3 domains (offline)...")
    domains = load_domains(llm.tokenizer, chunk_len=CHUNK_LEN,
                           train_chunks=TRAIN_CHUNKS, test_chunks=TEST_CHUNKS)
    order = " -> ".join(DOMAIN_ORDER)
    log(f"Domains: {order}  ({TRAIN_CHUNKS} train / {TEST_CHUNKS} test chunks each)")

    base = eval_row(llm, domains)
    log(f"Base ppl (frozen model): " +
        "  ".join(f"{n}={p:.1f}" for n, p in zip(DOMAIN_ORDER, base)))

    results = {"base_ppl": base, "domain_order": DOMAIN_ORDER, "methods": {}}
    summaries = {}
    for mode in ["naive", "joint", "ewc", "er", "stc_nodream", "stc_full"]:
        log(f"== {mode} ==")
        t0 = time.perf_counter()
        r = run_method(llm, adapter, domains, mode)
        dt = time.perf_counter() - t0
        results["methods"][mode] = r
        if mode == "joint":
            log(f"  avg final ppl (upper bound) = {r['avg_final_ppl']:.2f}  "
                f"[{dt:.0f}s]")
            summaries[mode] = {"avg_final_ppl": r["avg_final_ppl"]}
            continue
        s = compute_cl_summary(r["R"], base=base)
        summaries[mode] = s
        extra = ""
        if "replays" in r:
            extra = f" replays={r['replays']}"
        if "captures" in r:
            extra = f" captures={r['captures']}"
        log(f"  avg_final_ppl={s['avg_final_ppl']:.2f}  "
            f"avg_learned_ppl={s['avg_learned_ppl']:.2f}  "
            f"BWT(forgetting)={s['backward_transfer']:+.2f} "
            f"({s['forgetting_pct']:+.1f}%)  FWT={s['forward_transfer']:+.2f}  "
            f"[{extra} {dt:.0f}s]")

    log("== R-matrices (rows = after training domain i; cols = ppl per domain) ==")
    for mode in ["naive", "ewc", "er", "stc_nodream", "stc_full"]:
        R = np.array(results["methods"][mode]["R"])
        log(f"  {mode}:")
        for i, name in enumerate(DOMAIN_ORDER):
            log("    after " + name.ljust(6) + " " +
                "  ".join(f"{DOMAIN_ORDER[j]}={R[i, j]:7.1f}" for j in range(len(DOMAIN_ORDER))))

    log("== Verdict: forgetting (BWT, lower=better) ==")
    for mode in ["naive", "ewc", "er", "stc_nodream", "stc_full"]:
        s = summaries[mode]
        log(f"  {mode:12s} BWT={s['backward_transfer']:+7.2f} ppl "
            f"({s['forgetting_pct']:+6.1f}%)   final avg ppl={s['avg_final_ppl']:.1f}")
    log(f"  joint (ceiling) avg ppl={summaries['joint']['avg_final_ppl']:.1f}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(
        {"results": results, "summaries": summaries}, indent=2), encoding="utf-8")
    log(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
