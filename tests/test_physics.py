import numpy as np
import pytest
from merraflow.physics import project_precip, group_sum, budget_error, transform_target, inverse_target, native_groups
from merraflow.metrics import crps_ensemble, continuous, fss, rank_histogram


def test_conservation_dry_wet_and_zero_proposal():
    rng = np.random.default_rng(10)
    area = rng.uniform(1e6, 1e7, (19, 23))
    groups = np.arange(19*23).reshape(19, 23) % 17
    ref = rng.uniform(0, 20, groups.shape)
    ref[groups == 0] = 0
    proposal = rng.uniform(-3, 10, groups.shape)
    proposal[groups == 1] = 0  # force fallback for wet source / dry generated group
    projected = project_precip(proposal, ref, area, groups, dry_threshold=.1)
    assert (projected >= 0).all()
    assert np.all(projected[groups == 0] == 0)
    np.testing.assert_allclose(group_sum(projected, area, groups), group_sum(ref, area, groups), rtol=1e-6, atol=1e-5)
    assert budget_error(projected, ref, area, groups)['max_relative_wet'] < 1e-6
    np.testing.assert_allclose(project_precip(projected, ref, area, groups), projected, rtol=1e-6)


def test_projection_rejects_nan_and_negative_area():
    x = np.ones((4, 4))
    g = np.zeros((4, 4), int)
    with pytest.raises(ValueError):
        project_precip(x*np.nan, x, x, g)
    with pytest.raises(ValueError):
        project_precip(x, x, -x, g)


def test_transforms_round_trip_dry_and_extreme():
    x = np.array([[[280, 310]], [[0, 250]], [[70000, 105000]], [[0, 90]]], dtype='float32')
    np.testing.assert_allclose(inverse_target(transform_target(x)), x, rtol=1e-6)


def test_native_boundaries_and_non_square_grid():
    lat, lon = np.array([30., 31., 32.]), np.array([-100., -99., -98., -97.])
    y, x = np.meshgrid([30.1, 30.9], [-99.9, -98.6, -97.2], indexing='ij')
    groups, source = native_groups(lat, lon, y, x)
    np.testing.assert_array_equal(source[groups], [[0, 1, 3], [4, 5, 7]])
    with pytest.raises(ValueError):
        native_groups(lat[::-1], lon, y, x)


def test_crps_matches_pairwise_definition():
    rng = np.random.default_rng(4)
    members, truth = rng.normal(size=(7, 5, 6)), rng.normal(size=(5, 6))
    expected = np.mean(abs(members-truth), axis=0)-.5*np.mean(abs(members[:, None]-members[None]), axis=(0, 1))
    np.testing.assert_allclose(crps_ensemble(members, truth), expected, atol=1e-12)
    np.testing.assert_allclose(crps_ensemble(members[:1], truth), abs(members[0]-truth))


def test_scores_perfect_and_all_dry():
    x = np.arange(81).reshape(9, 9).astype(float)
    result = continuous(np.stack([x, x]), x, np.ones_like(x))
    assert result['rmse'] == result['crps'] == 0
    assert result['correlation'] == pytest.approx(1)
    assert fss(x, x, 30, 5, np.ones_like(x)) == 1
    assert fss(x*0, x*0, 1, 5, np.ones_like(x)) is None
    assert fss(x, x, 1, 17, np.ones_like(x)) is None
    hist = rank_histogram(np.zeros((3, 100, 100)), np.zeros((100, 100)), np.ones((100, 100)))
    np.testing.assert_allclose(hist, .25, atol=.02)
