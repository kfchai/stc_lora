"""Continual-learning summary metrics (perplexity-based BWT/FWT)."""

from __future__ import annotations

import numpy as np


def compute_cl_summary(
    R: list[list[float]] | np.ndarray,
    base: list[float] | np.ndarray | None = None,
) -> dict:
    """Continual-learning summary from a task-by-task PERPLEXITY matrix.

    Standard task-incremental evaluation (Lopez-Paz & Ranzato 2017, GEM),
    adapted to perplexity (lower = better) instead of accuracy (higher =
    better). R has shape (T, T): R[i][j] is perplexity on task j AFTER the
    model has finished training on task i. The diagonal R[i][i] is the model
    right after learning task i; the last row R[T-1][:] is the final model.

    Args:
        R: (T, T) perplexity matrix, row i = eval on all tasks after task i.
        base: optional length-T vector of perplexity on each task BEFORE any
              training (the frozen model). Required for forward transfer.

    Returns:
        avg_final_ppl: mean of the last row -- overall capability at the end.
        avg_learned_ppl: mean of the diagonal -- capability right after
                         learning each task (before later tasks interfere).
        backward_transfer: mean_{j<T-1} (R[T-1][j] - R[j][j]). In perplexity
                           units, POSITIVE = forgetting. The headline number.
        forgetting_pct: backward_transfer as a % of avg_learned_ppl.
        per_task_forgetting: R[T-1][j] - R[j][j] for each earlier task j.
        forward_transfer: mean_{j>=1} (base[j] - R[j-1][j]). POSITIVE =
                          earlier tasks made task j easier before it was
                          trained. None if base not given.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square (T,T); got {R.shape}")
    T = R.shape[0]

    diag = np.array([R[i, i] for i in range(T)])
    last = R[T - 1]

    avg_final = float(last.mean())
    avg_learned = float(diag.mean())

    if T > 1:
        per_task_forget = [float(R[T - 1, j] - R[j, j]) for j in range(T - 1)]
        bwt = float(np.mean(per_task_forget))
    else:
        per_task_forget = []
        bwt = 0.0
    forgetting_pct = 100.0 * bwt / max(avg_learned, 1e-9)

    fwt = None
    if base is not None and T > 1:
        base = np.asarray(base, dtype=float)
        fwt = float(np.mean([base[j] - R[j - 1, j] for j in range(1, T)]))

    return {
        "avg_final_ppl": avg_final,
        "avg_learned_ppl": avg_learned,
        "backward_transfer": bwt,
        "forgetting_pct": forgetting_pct,
        "per_task_forgetting": per_task_forget,
        "forward_transfer": fwt,
    }
