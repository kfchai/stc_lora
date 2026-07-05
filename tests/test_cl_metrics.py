"""Continual-learning summary metrics (BWT/FWT from a perplexity R-matrix)."""

from __future__ import annotations

import numpy as np
import pytest

from stc_lora.metrics import compute_cl_summary


def test_no_forgetting_diagonal_equals_last_row():
    # Perfect retention: final row equals the diagonal.
    R = [[10.0, 99.0, 99.0],
         [10.0, 12.0, 99.0],
         [10.0, 12.0, 14.0]]
    s = compute_cl_summary(R)
    assert s["backward_transfer"] == pytest.approx(0.0)
    assert s["forgetting_pct"] == pytest.approx(0.0)
    assert s["avg_final_ppl"] == pytest.approx((10 + 12 + 14) / 3)


def test_forgetting_is_positive_when_old_tasks_degrade():
    # Task 0 learned at ppl 10, but by the end it degraded to 30.
    R = [[10.0, 99.0, 99.0],
         [20.0, 12.0, 99.0],
         [30.0, 25.0, 14.0]]
    s = compute_cl_summary(R)
    # per-task forgetting: task0 = 30-10 = 20, task1 = 25-12 = 13
    assert s["per_task_forgetting"] == pytest.approx([20.0, 13.0])
    assert s["backward_transfer"] == pytest.approx(16.5)
    assert s["forgetting_pct"] > 0


def test_forward_transfer_positive_when_base_higher():
    R = [[10.0, 40.0, 90.0],
         [20.0, 12.0, 80.0],
         [30.0, 25.0, 14.0]]
    base = [50.0, 50.0, 100.0]  # frozen-model ppl before any training
    s = compute_cl_summary(R, base=base)
    # FWT: task1 seen before training = R[0,1]=40 vs base 50 -> +10
    #      task2 seen before training = R[1,2]=80 vs base 100 -> +20
    assert s["forward_transfer"] == pytest.approx(15.0)


def test_single_task_is_degenerate_safe():
    s = compute_cl_summary([[7.0]])
    assert s["backward_transfer"] == 0.0
    assert s["avg_final_ppl"] == pytest.approx(7.0)


def test_non_square_raises():
    with pytest.raises(ValueError):
        compute_cl_summary([[1.0, 2.0]])
