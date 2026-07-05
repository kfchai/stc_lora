"""Downstream forgetting: does sequential learning destroy CAPABILITY, or just
perplexity?

Reviewer-critical control for the STC-LoRA paper: BWT measured in perplexity
could in principle be calibration drift. Here we measure two task metrics on
the SAME streams and protocol as the main benchmark:

  1. AG News zero-shot topic classification (a real labeled downstream task):
     accuracy of picking the correct topic by label-word log-likelihood.
     Its decline after the stream moves on to wiki -> Shakespeare is genuine
     capability forgetting.
  2. Top-1 next-token accuracy per domain (task accuracy, not calibration).

Implementation: monkeypatch cl_benchmark.eval_row so every per-domain
checkpoint of the UNCHANGED protocol also records the task metrics.

Run:  MODEL=Qwen/Qwen2.5-0.5B DTYPE=float32 python -m experiments.downstream_forgetting
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

import experiments.cl_benchmark as cb
from stc_lora import STCLoRA, STCLoRAConfig
from stc_lora.backend import HFCausalLM

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
DTYPE = os.environ.get("DTYPE", "float32")
PERM_LR = 0.05
N_AG = 150                     # AG News test articles for the accuracy probe
METHODS = ["naive", "ewc", "er", "stc_frozen", "slow_stc"]
LABELS = [" world news", " sports", " business", " science and technology"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_agnews_test(n=N_AG):
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split=f"test[:{n}]")
    return [(t[:400], int(l)) for t, l in zip(ds["text"], ds["label"])]


@torch.no_grad()
def label_logprob(llm, prompt: str, label: str) -> float:
    """Sum of label-token logprobs given the prompt."""
    tok = llm.tokenizer
    p_ids = tok(prompt, return_tensors="pt").input_ids.to(llm.device)
    l_ids = tok(label, add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(llm.device)
    ids = torch.cat([p_ids, l_ids], dim=1)
    logits = llm.model(ids).logits[0]
    lp = torch.log_softmax(logits[:-1].float(), dim=-1)
    span = range(p_ids.shape[1] - 1, ids.shape[1] - 1)
    return float(sum(lp[i, ids[0, i + 1]] for i in span))


@torch.no_grad()
def agnews_accuracy(llm, items) -> float:
    correct = 0
    for text, label in items:
        prompt = f"Article: {text}\n\nThis article is about"
        scores = [label_logprob(llm, prompt, lab) for lab in LABELS]
        correct += int(np.argmax(scores)) == label
    return correct / len(items)


@torch.no_grad()
def token_accuracy(llm, chunks) -> float:
    """Mean top-1 next-token accuracy over a domain's held-out chunks."""
    hit = tot = 0
    for c in chunks:
        ids = torch.tensor([c], device=llm.device)
        pred = llm.model(ids).logits[0, :-1].argmax(-1)
        hit += int((pred == ids[0, 1:]).sum())
        tot += ids.shape[1] - 1
    return hit / max(1, tot)


def main() -> None:
    log(f"Loading {MODEL} ({DTYPE})...")
    llm = HFCausalLM(MODEL, dtype=DTYPE)
    adapter = STCLoRA(llm.model, STCLoRAConfig(
        rank=8, alpha=16.0, base_lr=cb.BASE_LR, capture_fraction=0.6,
    ))
    domains = cb.load_domains(llm.tokenizer, chunk_len=cb.CHUNK_LEN,
                              train_chunks=cb.TRAIN_CHUNKS,
                              test_chunks=cb.TEST_CHUNKS)
    ag = load_agnews_test()

    def task_metrics():
        return {"agnews_acc": agnews_accuracy(llm, ag),
                "tok_acc": [token_accuracy(llm, d.test_chunks)
                            for d in domains]}

    base = task_metrics()
    log(f"BASE: agnews={base['agnews_acc']:.3f}  tok_acc=" +
        " ".join(f"{d.name}={a:.3f}" for d, a in zip(domains, base["tok_acc"])))

    # record task metrics at every per-domain checkpoint of the protocol
    records: list = []
    orig_eval = cb.eval_row

    def patched_eval(llm_, doms):
        row = orig_eval(llm_, doms)
        records.append(task_metrics())
        return row

    results = {"base": base, "methods": {}}
    for method in METHODS:
        records.clear()
        cb.eval_row = orig_eval           # baseline rows without task metrics
        adapter.set_perm_plastic(PERM_LR if method == "slow_stc" else 0.0)
        mode = "stc_full" if method in ("stc_frozen", "slow_stc") else method
        cb.eval_row = patched_eval
        t0 = time.perf_counter()
        cb.run_method(llm, adapter, domains, mode, seed=0)
        cb.eval_row = orig_eval
        # records[k] = after finishing domain k (news, wiki, shake)
        after_news, after_all = records[0], records[-1]
        drop_ag = 100 * (after_news["agnews_acc"] - after_all["agnews_acc"]) \
            / max(1e-9, after_news["agnews_acc"])
        drop_tok = 100 * (after_news["tok_acc"][0] - after_all["tok_acc"][0]) \
            / max(1e-9, after_news["tok_acc"][0])
        results["methods"][method] = {
            "agnews_after_learning_news": after_news["agnews_acc"],
            "agnews_after_full_stream": after_all["agnews_acc"],
            "agnews_forgetting_pct": drop_ag,
            "news_tokacc_after_news": after_news["tok_acc"][0],
            "news_tokacc_after_full_stream": after_all["tok_acc"][0],
            "news_tokacc_forgetting_pct": drop_tok,
            "records": records.copy(),
        }
        log(f"  {method:11s} agnews {after_news['agnews_acc']:.3f}->"
            f"{after_all['agnews_acc']:.3f} ({drop_ag:+.1f}%)   "
            f"news tok-acc {after_news['tok_acc'][0]:.3f}->"
            f"{after_all['tok_acc'][0]:.3f} ({drop_tok:+.1f}%)  "
            f"[{time.perf_counter()-t0:.0f}s]")
    adapter.set_perm_plastic(0.0)

    log("== DOWNSTREAM FORGETTING (drop after stream moves past news) ==")
    for m, v in results["methods"].items():
        log(f"  {m:11s} agnews {v['agnews_forgetting_pct']:+6.1f}%   "
            f"news-token-acc {v['news_tokacc_forgetting_pct']:+6.1f}%")
    out = f"outputs/downstream_{MODEL.split('/')[-1]}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
