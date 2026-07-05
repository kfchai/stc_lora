"""STC-LoRA: continual learning for a frozen LLM via synaptic tagging & capture.

The differentiator of Track A. A pretrained transformer supplies fluency;
this adapter gives it the thing LLMs structurally lack — the ability to keep
learning after deployment, from single exposures, without catastrophic
forgetting.

The STC split-weight idea maps almost one-to-one onto LoRA:

    effective_weight = W_base  +  perm_delta  +  scaling * (B_tag @ A_tag)
                       └frozen┘  └consolidated┘ └── plastic, decaying ──┘

  - W_base:     the pretrained weight. Never touched.
  - tagged:     a low-rank LoRA pair (A_tag, B_tag). Gradient-updated on each
                exposure, learning rate GATED by the cerebellum's neuromod
                signal (surprise x coherence). Decays toward zero every step —
                un-reinforced changes are forgotten (this is what prevents the
                noise of one-off inputs from sticking).
  - perm_delta: a consolidated dense delta. Grows ONLY on a capture event,
                when neuromod exceeds a threshold (a genuinely significant,
                repeated, or high-reward exposure). Does not decay — this is
                long-term memory, and because it is added to the base rather
                than overwriting it, prior knowledge is preserved.

Biological basis: Frey & Morris (1997) synaptic tagging & capture. A synapse
is transiently "tagged" by activity (the plastic LoRA); only if a consolidation
signal (neuromodulator) arrives while the tag is present does the change become
permanent (the capture merge). Everything else decays.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class STCLoRAConfig:
    """Configuration for the STC-LoRA continual learner."""

    rank: int = 8
    alpha: float = 16.0                 # LoRA scaling = alpha / rank
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    base_lr: float = 5e-3               # Peak plastic LR (before neuromod gating)
    decay_rate: float = 0.05            # Per-step decay of the tagged delta
    capture_threshold: float = 0.6      # neuromod above this -> consolidate
    capture_fraction: float = 0.5       # Fraction of tag mass moved to permanent
    grad_clip: float = 1.0
    min_plasticity: float = 0.0         # Floor on neuromod-gated LR multiplier
    perm_lr_ratio: float = 0.0          # If >0, the permanent store is SLOWLY
                                        # plastic: perm_delta is refined by
                                        # gradient at base_lr * perm_lr_ratio,
                                        # so consolidated knowledge keeps
                                        # improving instead of being frozen at
                                        # capture-time quality. 0 = frozen
                                        # (original synaptic-capture behavior).


class STCLoRALinear(nn.Module):
    """A frozen Linear wrapped with a tagged (plastic) + permanent LoRA delta.

    Replaces a base ``nn.Linear`` in-place inside the host model. The base
    weight is frozen; all learning happens in the tagged pair and is
    consolidated into ``perm_delta`` on capture.
    """

    def __init__(self, base: nn.Linear, config: STCLoRAConfig):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.config = config
        in_f = base.in_features
        out_f = base.out_features
        r = config.rank
        self.scaling = config.alpha / r
        dev = base.weight.device
        dtype = base.weight.dtype

        # Tagged (plastic) low-rank pair. B starts at zero so the initial
        # delta is exactly zero — the wrapped model behaves identically to
        # the base until it learns something.
        self.A_tag = nn.Parameter(torch.randn(r, in_f, device=dev, dtype=dtype) * 0.01)
        self.B_tag = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=dtype))

        # Snapshot A_tag's initialization so reset_all() can restore the layer
        # to its exact starting state (B_tag zeros already give a zero delta,
        # but A_tag drifts under gradients — restoring it makes A/B arms that
        # reset between runs start identically).
        self.register_buffer("_A_tag_init", self.A_tag.detach().clone())

        # Permanent consolidated delta (dense). Grows by capture, and — when
        # perm_lr_ratio > 0 — is ALSO refined slowly by gradient so consolidated
        # knowledge keeps improving instead of freezing at capture-time quality.
        # A Parameter (trainable iff plastic); frozen mode leaves requires_grad
        # False so it behaves exactly like the original buffer.
        self._perm_plastic = config.perm_lr_ratio > 0
        self.perm_delta = nn.Parameter(
            torch.zeros(out_f, in_f, device=dev, dtype=dtype),
            requires_grad=self._perm_plastic,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # Permanent consolidated delta. When plastic, always include it so it
        # stays in the autograd graph (else it could never leave zero); when
        # frozen, skip the dense matmul while it is still all-zero.
        if self._perm_plastic or self.perm_delta.abs().sum() > 0:
            out = out + F.linear(x, self.perm_delta)
        # Tagged plastic delta: x -> A_tag -> B_tag
        tagged = F.linear(F.linear(x, self.A_tag), self.B_tag)
        return out + self.scaling * tagged

    @torch.no_grad()
    def decay(self) -> None:
        """Shrink the tagged delta toward zero (forget un-consolidated change).

        Decay the output-side factor only, so the effective delta shrinks
        linearly rather than quadratically.
        """
        self.B_tag.mul_(1.0 - self.config.decay_rate)

    @torch.no_grad()
    def capture(self, strength: float) -> float:
        """Consolidate part of the tagged delta into the permanent delta.

        Returns the L1 mass moved (for monitoring). Moves ``capture_fraction``
        of the current tagged contribution into perm_delta and removes that
        much from the tag, so the same change stops decaying and becomes
        permanent memory.
        """
        frac = self.config.capture_fraction * strength
        if frac <= 0:
            return 0.0
        # Current tagged delta as a dense matrix: scaling * B_tag @ A_tag
        delta = self.scaling * (self.B_tag @ self.A_tag)
        moved = frac * delta
        self.perm_delta.add_(moved)
        self.B_tag.mul_(1.0 - frac)  # remove the captured mass from the tag
        return float(moved.abs().sum())

    @torch.no_grad()
    def tag_norm(self) -> float:
        return float((self.scaling * (self.B_tag @ self.A_tag)).norm())


class STCLoRA:
    """Attaches STC-LoRA layers to a HuggingFace causal LM and drives the
    tagged/decay/capture dynamics from the cerebellum's neuromod signal.

    Usage:
        stc = STCLoRA(hf_backend.model, config)
        # per incoming exposure, with cerebellum signals already computed:
        stc.learn(input_ids, neuromod=nm)   # gated update + decay + capture
    """

    def __init__(self, model: nn.Module, config: STCLoRAConfig | None = None):
        self.model = model
        self.config = config or STCLoRAConfig()
        self.layers: list[STCLoRALinear] = []
        self._inject()
        self._opt = torch.optim.SGD(self.trainable_parameters(), lr=1.0)
        # Slow optimizer for the permanent store (only when it is plastic).
        self.perm_active = self.config.perm_lr_ratio > 0
        self._perm_opt = (
            torch.optim.SGD([l.perm_delta for l in self.layers], lr=1.0)
            if self.perm_active else None
        )
        self._step = 0
        self._n_captures = 0

    # -- injection ----------------------------------------------------

    def _inject(self) -> None:
        """Replace every target nn.Linear with an STCLoRALinear wrapper."""
        targets = self.config.target_modules
        replaced = 0
        for name, module in list(self.model.named_modules()):
            for child_name, child in list(module.named_children()):
                if (isinstance(child, nn.Linear)
                        and any(t == child_name for t in targets)):
                    wrapper = STCLoRALinear(child, self.config)
                    setattr(module, child_name, wrapper)
                    self.layers.append(wrapper)
                    replaced += 1
        if replaced == 0:
            raise ValueError(
                f"No target modules {targets} found in model. "
                f"Check module names via model.named_modules()."
            )

    def trainable_parameters(self):
        for layer in self.layers:
            yield layer.A_tag
            yield layer.B_tag

    def set_perm_plastic(self, perm_lr_ratio: float) -> None:
        """Toggle the permanent store between frozen (0) and slowly plastic.

        Lets one injected adapter switch modes between experiment arms without
        re-wrapping the model. Rebuilds the slow optimizer as needed.
        """
        self.config.perm_lr_ratio = perm_lr_ratio
        plastic = perm_lr_ratio > 0
        self.perm_active = plastic
        for layer in self.layers:
            layer._perm_plastic = plastic
            layer.perm_delta.requires_grad_(plastic)
        self._perm_opt = (
            torch.optim.SGD([l.perm_delta for l in self.layers], lr=1.0)
            if plastic else None
        )

    # -- the continual-learning step ----------------------------------

    def plasticity(self, neuromod: float) -> float:
        """Map the cerebellum neuromod signal to an LR multiplier."""
        return max(self.config.min_plasticity, float(neuromod))

    def learn(self, input_ids: torch.Tensor, neuromod: float,
              attention_mask: torch.Tensor | None = None) -> dict:
        """One surprise-gated continual-learning step on a single exposure.

        1. Forward + causal-LM loss (teacher forced on the exposure itself).
        2. Backward; SGD step on tagged params with LR = base_lr * plasticity.
        3. Decay all tagged deltas.
        4. If neuromod exceeds capture_threshold, consolidate into permanent.

        Args:
            input_ids: (1, T) or (T,) token ids of the exposure.
            neuromod: cerebellum neuromodulatory signal in [0, 1].
            attention_mask: optional mask matching input_ids.

        Returns dict with loss, plasticity, captured (bool), moved_mass.
        """
        cfg = self.config
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        plast = self.plasticity(neuromod)
        lr = cfg.base_lr * plast

        self.model.train()
        self._opt.zero_grad(set_to_none=True)
        if self._perm_opt is not None:
            self._perm_opt.zero_grad(set_to_none=True)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         labels=input_ids)
        loss = out.loss
        moved_mass = 0.0
        captured = False

        if lr > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.trainable_parameters()), cfg.grad_clip
            )
            for g in self._opt.param_groups:
                g["lr"] = lr
            self._opt.step()
            # Slow, ungated refinement of the permanent store (fast/slow weights)
            if self._perm_opt is not None and self.perm_active:
                perm_params = [l.perm_delta for l in self.layers]
                torch.nn.utils.clip_grad_norm_(perm_params, cfg.grad_clip)
                perm_lr = cfg.base_lr * cfg.perm_lr_ratio
                for g in self._perm_opt.param_groups:
                    g["lr"] = perm_lr
                self._perm_opt.step()

        # STC dynamics (no autograd)
        for layer in self.layers:
            layer.decay()
        if neuromod > cfg.capture_threshold:
            strength = min(1.0, (neuromod - cfg.capture_threshold)
                           / max(1e-6, 1.0 - cfg.capture_threshold))
            for layer in self.layers:
                moved_mass += layer.capture(strength)
            captured = True
            self._n_captures += 1

        self.model.eval()
        self._step += 1
        return {
            "loss": float(loss.detach()),
            "plasticity": plast,
            "lr": lr,
            "captured": captured,
            "moved_mass": moved_mass,
        }

    # -- low-level hooks for baselines (plain LoRA, EWC, replay) -------
    #
    # These drive the SAME tagged LoRA weights but with a caller-supplied
    # loss and NO STC dynamics (no decay, no capture, no gating). They let
    # continual-learning baselines share this adapter's substrate so any
    # difference is the algorithm, not the implementation.

    def lm_loss(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Causal-LM loss on input_ids (teacher forced). Keeps the graph."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        self.model.train()
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         labels=input_ids)
        return out.loss

    def sgd_step(self, loss: torch.Tensor, lr: float) -> None:
        """One plain SGD step on the tagged params (no decay/capture).

        Used by the CL baselines (plain LoRA / EWC / replay). The permanent
        store is never stepped here; any perm grads are cleared so a plastic-
        perm adapter reused across methods doesn't leak gradients.
        """
        self._opt.zero_grad(set_to_none=True)
        if self._perm_opt is not None:
            self._perm_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.trainable_parameters()), self.config.grad_clip
        )
        for g in self._opt.param_groups:
            g["lr"] = lr
        self._opt.step()
        self.model.eval()
        self._step += 1

    # -- introspection ------------------------------------------------

    @torch.no_grad()
    def tag_norm(self) -> float:
        return sum(l.tag_norm() for l in self.layers)

    @torch.no_grad()
    def perm_norm(self) -> float:
        return sum(float(l.perm_delta.norm()) for l in self.layers)

    @property
    def n_captures(self) -> int:
        return self._n_captures

    @torch.no_grad()
    def reset_tags(self) -> None:
        """Clear all tagged deltas (permanent memory is kept)."""
        for layer in self.layers:
            layer.B_tag.zero_()

    @torch.no_grad()
    def reset_all(self) -> None:
        """Restore every layer to its initial state — model returns to base.

        Zeros both deltas and restores A_tag to its initialization, so runs
        that reset between them (e.g. A/B experiment arms) start identically.
        """
        for layer in self.layers:
            layer.B_tag.zero_()
            layer.perm_delta.zero_()
            layer.A_tag.copy_(layer._A_tag_init)
        self._n_captures = 0
        self._step = 0
