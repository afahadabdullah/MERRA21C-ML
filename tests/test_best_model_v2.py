"""Guard the diagnostic's split selection and event crop."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts'/'test_best_model_v2.py'
SPEC = importlib.util.spec_from_file_location('test_best_model_v2_script', SCRIPT)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def test_explicit_case_cannot_cross_validation_test_boundary():
    class Archive:
        index = {'entries': [
            {'id': '20251203_0030', 'time': '2025-12-03T00:30:00', 'split': 'val'},
            {'id': '20260203_0030', 'time': '2026-02-03T00:30:00', 'split': 'test'},
            {'id': '20260204_0030', 'time': '2026-02-04T00:30:00', 'split': 'test'},
        ]}

    with pytest.raises(ValueError, match='found 0'):
        DIAGNOSTIC.select_entries(Archive(), 'test', 3, ['20251203_0030'], 317)
    selected, meta = DIAGNOSTIC.select_entries(Archive(), 'val', 3, ['20251203_0030'], 317)
    assert [e['id'] for e in selected] == ['20251203_0030']
    assert meta['strategy'] == 'explicit timestamps'
    first, _ = DIAGNOSTIC.select_entries(Archive(), 'test', 2, None, 317)
    second, _ = DIAGNOSTIC.select_entries(Archive(), 'test', 2, None, 317)
    assert first == second and all(e['split'] == 'test' for e in first)
    with pytest.raises(ValueError, match='unique'):
        DIAGNOSTIC.select_entries(Archive(), 'test', 1, ['20260203_0030']*2, 317)


def test_zoom_stays_inside_small_domain_and_contains_rain_event():
    rain = np.zeros((24, 40), dtype='float32')
    rain[20:23, 35:39] = 8
    ys, xs = DIAGNOSTIC.event_window(rain, .4)
    assert 0 <= ys.start < 21 < ys.stop <= rain.shape[0]
    assert 0 <= xs.start < 37 < xs.stop <= rain.shape[1]


def test_include_date_selects_wettest_hour_then_four_other_test_hours():
    class Archive:
        static = {'area': np.ones((2, 2))}
        index = {'entries': [
            {'id': f'20260223_{hour:02d}30', 'time': f'2026-02-23T{hour:02d}:30:00', 'split': 'test'}
            for hour in (0, 6, 12)
        ] + [
            {'id': f'20260224_{hour:02d}30', 'time': f'2026-02-24T{hour:02d}:30:00', 'split': 'test'}
            for hour in (0, 6, 12, 18)
        ]}

        def array(self, entry, name):
            assert name == 'truth'
            field = np.zeros((5, 2, 2))
            field[1] = 10 if entry['id'] == '20260223_1230' else 1
            return field

    selected, meta = DIAGNOSTIC.select_entries(Archive(), 'test', 5, None, 317, '2026-02-23')
    assert len(selected) == len({e['id'] for e in selected}) == 5
    assert selected[0]['id'] == meta['event_id'] == '20260223_1230'
    assert meta['event_mean_precip_mm_h'] == 10
    assert all(e['split'] == 'test' for e in selected)
    assert sum(e['time'].startswith('2026-02-23') for e in selected) == 1
    with pytest.raises(ValueError, match='No test entries'):
        DIAGNOSTIC.select_entries(Archive(), 'test', 5, None, 317, '2026-02-25')
