"""Reservoir sampler is temporally uniform (domain-balanced).

The P1 fix: replace the recency-skewed reservoir with Vitter Algorithm R so
the replay buffer represents the whole stream, not just recent items.
"""

from __future__ import annotations

import numpy as np

from experiments.cl_benchmark import ReservoirSampler


def test_holds_at_most_cap():
    rng = np.random.default_rng(0)
    r = ReservoirSampler(cap=10, rng=rng)
    for i in range(500):
        r.offer(i)
    assert len(r.items) == 10
    assert r.seen == 500


def test_balanced_across_three_domains():
    # Stream: 300 items, 3 equal-size "domains" (0=news,1=wiki,2=shake) in
    # order. A recency-skewed buffer would over-represent domain 2; a uniform
    # reservoir should be roughly balanced.
    counts = np.zeros(3)
    trials = 400
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        r = ReservoirSampler(cap=30, rng=rng)
        for i in range(300):
            domain = i // 100
            r.offer(domain)
        for d in r.items:
            counts[d] += 1
    frac = counts / counts.sum()
    # Each domain should get ~1/3 of the buffer on average. Recency-skew would
    # push domain 2 well above 0.5; uniform keeps all three near 0.33.
    assert np.all(np.abs(frac - 1 / 3) < 0.06), frac.tolist()


def test_early_items_survive():
    # The very first item should still appear in the buffer a fair fraction of
    # the time (cap/n), unlike a recency-biased scheme where it is quickly lost.
    kept = 0
    trials = 2000
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        r = ReservoirSampler(cap=10, rng=rng)
        for i in range(200):
            r.offer(i)
        if 0 in r.items:
            kept += 1
    # Expected retention probability = cap/n = 10/200 = 0.05.
    assert 0.03 < kept / trials < 0.07, kept / trials
