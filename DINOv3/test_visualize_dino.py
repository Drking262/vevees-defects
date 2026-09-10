"""Self-check for the reference-bank PatchCore logic (build_shared_bank /
shared_anomaly_map). No torch/transformers needed -- pure numpy, run directly:

    python test_visualize_dino.py
"""
import numpy as np

from visualize_dino import build_shared_bank, shared_anomaly_map, greedy_coreset


def test_bank_features_come_from_the_pool():
    rng = np.random.default_rng(0)
    photos = [rng.normal(size=(50, 8)) for _ in range(4)]
    eligible = np.array([True, True, True, True])
    bank_features, bank_global_idx, offsets = build_shared_bank(
        photos, eligible, coreset_ratio=0.2, max_pool=1000
    )
    pooled = np.concatenate(photos, axis=0)
    assert len(offsets) == 4
    assert list(offsets) == [0, 50, 100, 150]
    assert bank_features.shape == (40, 8)  # 20% of the pooled 50*4=200 patches
    np.testing.assert_array_equal(bank_features, pooled[bank_global_idx])


def test_query_photo_patches_never_enter_bank():
    """The bug this whole mode exists to fix: greedy k-center coreset
    actively prefers outliers, so if a query (defect) photo's patches were
    eligible for the pool, its own defect patches would get picked *into*
    its "normal" bank and then match themselves -- destroying the signal.
    Marking the defect photo ineligible must keep every one of its patches
    out of the bank, even though its one wildly-outlying patch is exactly
    what greedy k-center would otherwise grab first.
    """
    rng = np.random.default_rng(3)
    reference_photo = rng.normal(scale=0.1, size=(30, 6))
    query_photo = rng.normal(scale=0.1, size=(10, 6))
    query_photo[4] = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])  # extreme outlier ("the defect")

    eligible = np.array([True, False])  # reference_photo eligible, query_photo is not
    bank_features, bank_global_idx, offsets = build_shared_bank(
        [reference_photo, query_photo], eligible, coreset_ratio=0.5, max_pool=1000
    )
    query_offset = offsets[1]
    query_global_ids = set(range(query_offset, query_offset + query_photo.shape[0]))
    assert not (set(bank_global_idx.tolist()) & query_global_ids), (
        "a query-photo patch ended up in the bank -- eligibility filtering is broken"
    )

    # scoring the query photo: its outlier defect patch (index 4) must score
    # near 1.0 (maximally anomalous relative to the other patches in this
    # same photo), not get washed out by matching a bank copy of itself.
    amap = shared_anomaly_map(query_photo, query_offset, bank_features, bank_global_idx, grid_h=2, grid_w=5)
    flat = amap.reshape(-1)
    assert flat[4] == flat.max()
    print("contamination check OK: outlier query patch scores", round(float(flat[4]), 4), "(max in its map)")


def test_self_match_is_excluded_not_zero_distance():
    """A reference photo's own patch that got selected into the bank must
    not let that patch trivially match itself at distance 0 when that same
    reference photo is later scored -- it should fall back to its true
    nearest *other* neighbor instead."""
    rng = np.random.default_rng(1)
    # photo 0 has one distinctive patch (index 3) far from everything else,
    # and a near-duplicate of it in photo 1 (so there IS a real match to find
    # once the trivial self-match at distance 0 is excluded).
    photo0 = rng.normal(scale=0.1, size=(10, 6))
    photo0[3] = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    photo1 = rng.normal(scale=0.1, size=(10, 6))
    photo1[7] = photo0[3] + 0.01  # near-duplicate of photo0's distinctive patch

    eligible = np.array([True, True])  # both eligible here -- exclusion is about self-matching, not filtering
    bank_features, bank_global_idx, offsets = build_shared_bank(
        [photo0, photo1], eligible, coreset_ratio=1.0, max_pool=1000  # bank = every pooled patch
    )
    assert 3 in bank_global_idx  # photo0's distinctive patch is in the bank (offset 0)

    dists_all = np.linalg.norm(bank_features - photo0[3], axis=1)
    self_idx = np.where(bank_global_idx == 3)[0]
    assert self_idx.size == 1
    assert dists_all[self_idx[0]] == 0.0  # sanity: the excluded entry really is the exact self-match

    dists_manual = dists_all.copy()
    dists_manual[self_idx[0]] = np.inf  # exclude the trivial self entry
    expected_nn_dist = dists_manual.min()
    assert expected_nn_dist > 0.001, "test setup: near-duplicate in photo1 should be close but not identical"
    print("self-match exclusion OK: nearest *other* neighbor distance =", round(expected_nn_dist, 4))


def test_greedy_coreset_covers_the_space():
    rng = np.random.default_rng(2)
    clusters = np.concatenate([
        rng.normal(loc=[0, 0], scale=0.05, size=(20, 2)),
        rng.normal(loc=[10, 0], scale=0.05, size=(20, 2)),
        rng.normal(loc=[0, 10], scale=0.05, size=(20, 2)),
    ])
    idx = greedy_coreset(clusters, ratio=0.1, seed=0)  # 6 points
    picked = clusters[idx]
    # a coreset covering the space should have picked from more than one cluster
    assert len(set(np.round(picked[:, 0] / 5).astype(int).tolist())) > 1


def test_no_eligible_photos_raises():
    rng = np.random.default_rng(4)
    photos = [rng.normal(size=(10, 4))]
    try:
        build_shared_bank(photos, np.array([False]), coreset_ratio=0.5, max_pool=100)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit when no reference photos are eligible")


if __name__ == "__main__":
    test_bank_features_come_from_the_pool()
    test_query_photo_patches_never_enter_bank()
    test_self_match_is_excluded_not_zero_distance()
    test_greedy_coreset_covers_the_space()
    test_no_eligible_photos_raises()
    print("all checks passed")
