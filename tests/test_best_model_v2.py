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
