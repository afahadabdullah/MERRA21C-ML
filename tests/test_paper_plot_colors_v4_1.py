"""Verify requested display scales without changing the underlying scores."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture
def plots():
    spec = importlib.util.spec_from_file_location('paper_plot_colors', Path('scripts/paper_plots_v4_1.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precipitation_scales_have_linear_five_unit_intervals(plots):
    cmap, norm = plots.rain_cmap()
    assert type(norm) is colors.Normalize
    np.testing.assert_allclose(np.diff(norm(plots.RAIN_LEVELS)), .1)
    np.testing.assert_allclose(cmap(norm(0)), [1, 1, 1, 1])
    for signed in (False, True):
        norm, ticks = plots.precip_error_scale(12.3, signed)
        np.testing.assert_allclose(np.diff(ticks), 5)
        np.testing.assert_allclose(np.diff(norm(ticks)), np.diff(norm(ticks))[0])
        assert norm.vmax >= 12.3 and (norm.vmin == -norm.vmax if signed else norm.vmin == 0)


def test_white_band_keeps_linear_values_and_distinguishes_missing_data(plots):
    base = plt.get_cmap('RdBu').with_extremes(bad='#e6e8eb')
    cmap, norm = plots.difference_scale(-80, 80, 5, base)
    values = np.array([-5, -2.5, 0, 2.5, 5.])
    original = values.copy()
    np.testing.assert_allclose(cmap(norm(values)), np.ones((5, 4)))
    np.testing.assert_allclose(np.diff(norm(values)), 2.5/160)
    assert not np.allclose(cmap(norm(-20)), [1, 1, 1, 1])
    assert not np.allclose(cmap(norm(20)), [1, 1, 1, 1])
    np.testing.assert_allclose(cmap(np.ma.masked), base.get_bad())
    np.testing.assert_array_equal(values, original)


def test_gain_figure_whitens_small_changes_but_retains_scores(plots, monkeypatch, tmp_path):
    truth = np.broadcast_to(np.array([280, 1, 100000, 2, 1, .008])[:, None, None], (6, 8, 12)).copy()
    change = np.array([.05, .05, 1, .05, .05, .00002])[:, None, None]
    case = SimpleNamespace(lat=np.zeros((8, 12)), area=np.ones((8, 12)), truth=truth,
                           coarse=truth+2*change, ensemble=np.stack([truth+change]*3))
    captured = []
    monkeypatch.setattr(plots, 'export', lambda fig, *args, **kwargs: captured.append(fig))
    result = plots.fig_gain(case, None, tmp_path, 72, 'gain')
    try:
        images = [ax.images[0] for ax in captured[0].axes if ax.images]
        for row, name in enumerate(plots.GAIN_ROWS):
            image = images[3*row+2]
            assert image.norm.vmin == -image.norm.vmax
            np.testing.assert_allclose(image.cmap(image.norm(0)), [1, 1, 1, 1])
        rain = images[3*plots.GAIN_ROWS.index('precip')+2]
        np.testing.assert_allclose(rain.cmap(rain.norm(rain.get_array())),
                                   np.ones((*rain.get_array().shape, 4)))
        assert result['numbers']['precip']['skill'] == pytest.approx(.5)
        assert result['numbers']['precip']['improved_area'] == 1
        assert result['numbers']['precip']['coarse_mae'] == pytest.approx(.1)
    finally:
        plt.close(captured[0])


def test_skill_maps_whiten_one_percent_without_changing_metrics(plots, monkeypatch, tmp_path):
    arrays = {}
    for name in plots.SKILL_MAP_ROWS:
        arrays.update({f'{name}_mae_coarse': np.ones((8, 12)),
                       f'{name}_crps': np.full((8, 12), .99),
                       f'{name}_se_coarse': np.ones((8, 12)),
                       f'{name}_se_mean': np.full((8, 12), .99**2)})
    meta = dict(n_hours=2, hours=['first', 'second'], members=16, complete=True)
    path = tmp_path/'maps.npz'
    np.savez(path, meta=np.asarray(json.dumps(meta)), **arrays)
    captured = []
    monkeypatch.setattr(plots, 'export', lambda fig, *args, **kwargs: captured.append(fig))
    result = plots.fig_skill_maps(path, tmp_path, 72, 'skill_maps')
    try:
        images = [ax.images[0] for ax in captured[0].axes if ax.images]
        for row, name in enumerate(plots.SKILL_MAP_ROWS):
            for col in (2, 3):
                image = images[4*row+col]
                assert (image.norm.vmin, image.norm.vmax) == (-80, 80)
                np.testing.assert_allclose(image.get_array(), 1)
                np.testing.assert_allclose(image.cmap(image.norm(image.get_array())),
                                           np.ones((*image.get_array().shape, 4)))
            assert result['numbers'][name]['skill'] == pytest.approx(.01)
            assert result['numbers'][name]['rmse_reduction'] == pytest.approx(.01)
    finally:
        plt.close(captured[0])


def test_scorecard_and_temporal_skill_share_symmetric_range(plots, monkeypatch, tmp_path):
    from test_evaluate_all_v4_1 import records_for_scores
    from merraflow.evaluate_all_v4_1 import aggregate_sources
    records = records_for_scores()
    summary = dict(sources=aggregate_sources(records))
    manifest = dict(entries=[dict(id=r['id'], time=r['time']) for r in records])
    for record in records:
        (tmp_path/(record['id']+'.json')).write_text(json.dumps(record))
    captured = []
    monkeypatch.setattr(plots, 'export', lambda fig, *args, **kwargs: captured.append(fig))
    try:
        plots.fig_scorecard(summary, tmp_path, 72, 'matrix')
        plots.fig_temporal(manifest, [r['id'] for r in records], tmp_path, tmp_path, 72, 'temporal')
        for fig in captured:
            image = next(ax.images[0] for ax in fig.axes if ax.images)
            assert (image.norm.vmin, image.norm.vmax) == (-80, 80)
            np.testing.assert_allclose(image.cmap(image.norm([-5, 0, 5])), np.ones((3, 4)))
    finally:
        for fig in captured:plt.close(fig)
