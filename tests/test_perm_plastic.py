"""Slow-plastic permanent store: perm_delta refines by gradient when enabled.

Uses a tiny stand-in model (a single Linear named 'q_proj') so gradients flow
without loading an LLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stc_lora import STCLoRA, STCLoRAConfig


class _Tiny(nn.Module):
    """Minimal model exposing a `q_proj` and an HF-like (loss) forward."""

    def __init__(self, d=16):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.head = nn.Linear(d, d)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        x = input_ids.float()
        y = self.head(torch.relu(self.q_proj(x)))
        loss = ((y - labels.float()) ** 2).mean()
        return type("O", (), {"loss": loss})()


def _cfg(**kw):
    return STCLoRAConfig(rank=4, alpha=8.0, target_modules=("q_proj",),
                         base_lr=0.1, **kw)


def test_frozen_perm_stays_zero_without_capture():
    torch.manual_seed(0)
    stc = STCLoRA(_Tiny(), _cfg(perm_lr_ratio=0.0, capture_threshold=1e9))
    x = torch.randn(1, 16)
    for _ in range(5):
        stc.learn(x, neuromod=0.5)
    # No capture, frozen perm -> permanent store never moves off zero.
    assert stc.perm_norm() == 0.0


def test_plastic_perm_refines_by_gradient():
    torch.manual_seed(0)
    stc = STCLoRA(_Tiny(), _cfg(perm_lr_ratio=0.5, capture_threshold=1e9))
    x = torch.randn(1, 16)
    for _ in range(10):
        stc.learn(x, neuromod=0.5)
    # Even with NO capture, the plastic permanent store grows via gradient.
    assert stc.perm_norm() > 0.0


def test_toggle_switches_mode():
    torch.manual_seed(0)
    stc = STCLoRA(_Tiny(), _cfg(perm_lr_ratio=0.0, capture_threshold=1e9))
    assert stc._perm_opt is None
    stc.set_perm_plastic(0.3)
    assert stc._perm_opt is not None and stc.perm_active
    x = torch.randn(1, 16)
    for _ in range(8):
        stc.learn(x, neuromod=0.5)
    assert stc.perm_norm() > 0.0
    stc.set_perm_plastic(0.0)
    assert stc._perm_opt is None and not stc.perm_active


def test_learn_still_returns_expected_keys():
    stc = STCLoRA(_Tiny(), _cfg(perm_lr_ratio=0.5))
    info = stc.learn(torch.randn(1, 16), neuromod=0.9)
    assert {"loss", "plasticity", "captured"} <= set(info)
