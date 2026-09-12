from collections import Counter
from copy import deepcopy
from datetime import datetime

import pytest

from merraflow.config_v2 import load_config_v2
from merraflow.prepare_v2 import preparation_months, requested_hours


def test_annual_split_boundaries_and_months_v2():
    cfg = load_config_v2('configs/discover_annual_v2.yaml')
    hours = dict(requested_hours(cfg))
    assert Counter(hours.values()) == {'train': 8760, 'val': 1440, 'test': 1368}
    assert hours[datetime(2024, 12, 1, 0, 30)] == 'train'
    assert hours[datetime(2025, 11, 30, 23, 30)] == 'train'
    assert datetime(2025, 12, 1, 0, 30) not in hours
    assert datetime(2025, 12, 2, 23, 30) not in hours
    assert hours[datetime(2025, 12, 3, 0, 30)] == 'val'
    assert hours[datetime(2026, 1, 31, 23, 30)] == 'val'
    assert datetime(2026, 2, 1, 0, 30) not in hours
    assert datetime(2026, 2, 2, 23, 30) not in hours
    assert hours[datetime(2026, 2, 3, 0, 30)] == 'test'
    assert hours[datetime(2026, 3, 31, 23, 30)] == 'test'
    assert len({t.month for t, split in hours.items() if split == 'train'}) == 12
    assert preparation_months(cfg) == ['2024-12'] + [f'2025-{m:02d}' for m in range(1, 13)] + [
        '2026-01', '2026-02', '2026-03']


def test_month_schedule_skips_unrequested_months_v2():
    cfg = deepcopy(load_config_v2('configs/discover_annual_v2.yaml'))
    cfg['data']['splits']['val'] = ['2026-01-01', '2026-02-01']
    months = preparation_months(cfg)
    assert '2025-12' not in months and '2024-12' in months
    cfg['data']['splits']['val'][0] = '2025-11-01'
    with pytest.raises(ValueError, match='overlap'):
        preparation_months(cfg)
