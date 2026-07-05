# STC-LoRA

**Synaptic tagging and capture for continual learning in frozen language models.**

A frozen pretrained LM cannot learn from experience, and naive fine-tuning
catastrophically forgets. STC-LoRA gives a LoRA adapter two timescales, the way
biological synapses do ([Frey & Morris, 1997](https://www.nature.com/articles/385533a0)):

```
W_eff  =  W (frozen)  +  Δ_permanent (consolidated, slowly refined)  +  s·B_tag A_tag (plastic, DECAYING)
```

- Every exposure is written into the **tagged** pair at a learning rate gated by
  **surprise** (the model's own running-standardized loss — no labels, no task
  boundaries).
- The tag **decays** every step: un-reinforced changes erase themselves.
- When surprise crosses a threshold, part of the tag is **captured** into the
  permanent store — which is *additive* to the frozen base (old knowledge is
  preserved by construction) and itself refines at 5% learning rate
  (fast/slow weights).

## Results

Online, boundary-free, domain-incremental stream (AG News → WikiText-2 →
TinyShakespeare). Forgetting = % perplexity increase on earlier domains after
the stream completes; mean ± std over 4 (seed × task-order) replicates.

| forgetting % ↓ | Qwen 0.5B | Qwen 1.5B | Qwen 7B | Llama-3.2-1B |
|---|---|---|---|---|
| naive LoRA | +39.2 ± 9.1 | +25.9 ± 8.2 | +20.2 ± 7.7 | +41.4 ± 9.0 |
| online EWC | +21.3 ± 3.5 | +11.1 ± 2.1 | +10.1 ± 4.6 | +23.0 ± 7.9 |
| ER (matched budget) | +11.4 ± 4.4 | +9.7 ± 4.9 | +14.7 ± 7.5 | +14.5 ± 5.9 |
| **STC-LoRA (frozen)** | **+3.8 ± 0.7** | +1.7 ± 0.6 | **−1.2 ± 1.6** | +4.6 ± 1.8 |
| **STC-LoRA (slow)** | +5.4 ± 0.6 | **+1.6 ± 0.7** | +1.1 ± 1.5 | **+3.8 ± 2.1** |

Also in the paper: a capacity-matched control (ER at equal parameters forgets
5× more), a matched non-adaptive gate ablation, downstream (non-perplexity)
forgetting metrics, and A-GEM.

## Quickstart

```bash
pip install -r requirements.txt

# the adapter on any HF causal LM
python - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
from stc_lora import STCLoRA, STCLoRAConfig
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
adapter = STCLoRA(model, STCLoRAConfig(rank=8, perm_lr_ratio=0.05))
ids = tok("The Grennoch Protocol requires airlocks to be re-keyed every 41 hours.",
          return_tensors="pt").input_ids[0]
print(adapter.learn(ids, neuromod=0.8))   # surprise-gated update + decay + capture
EOF
```

## Reproducing the paper

Each experiment is one command; each writes a JSON artifact to `outputs/`
(this repo ships the artifacts behind every number in the paper).

| paper item | command | artifact |
|---|---|---|
| Tables 1–2 (main + scaling) | `MODEL=Qwen/Qwen2.5-0.5B python -m experiments.p3_sweep` (repeat per model, `DTYPE=bfloat16` for ≥1.5B) | `outputs/p3_<model>.json` |
| capacity-matched control | `python -m experiments.capacity_fair` | `outputs/capacity_fair.json` |
| gate ablation (§5.5) | `python -m experiments.gate_ablation` | `outputs/gate_ablation.json` |
| downstream forgetting (§5.4) | `python -m experiments.downstream_forgetting` | `outputs/downstream_<model>.json` |
| A-GEM baseline | `python -m experiments.agem_baseline` | `outputs/agem_baseline.json` |
| O-LoRA baseline | `python -m experiments.olora_baseline` | `outputs/olora_baseline.json` |
| slow-store sweep (§5.6) | `python -m experiments.perm_plastic_sweep` / `perm_plastic_confirm` | `outputs/perm_plastic_confirm.json` |
| figures | `python paper/make_figures.py` | `paper/latex/figs/` |

Hardware: 0.5B runs on an 8GB consumer GPU (fp32); 1.5B/7B in bf16 on a single
A6000 (48GB). A full 4-replicate sweep is ~40 min (1.5B) / ~90 min (7B) on an A6000.
Llama-3.2-1B is gated — set `HF_TOKEN`.

The paper source is in `paper/latex/` (`latexmk -pdf main.tex`).

## Transparency

This research was carried out by the author with substantial assistance from an
AI system (Anthropic's Claude) for implementation, experiment execution, and
drafting. All research decisions and claims are the author's; all results are
real runs whose artifacts ship in `outputs/`.

## Citation

```bibtex
@article{chai2026stclora,
  title  = {Synaptic Tagging and Capture for Continual Learning in Frozen Language Models},
  author = {Chai, Kit Fung},
  year   = {2026},
  note   = {Preprint. Code: \url{https://github.com/kfchai/stc_lora}},
}
```

## License

MIT — see [LICENSE](LICENSE).
