"""Five-channel v2 transforms. No precipitation conservation projection."""
import numpy as np

TARGETS_V2 = ('t2m', 'precip', 'ps', 'u10m', 'v10m')
UNITS_V2 = ('K', 'mm h-1', 'Pa', 'm s-1', 'm s-1')


def transform_v2(x, precip_scale):
    x = np.asarray(x, dtype='float32').copy()
    x[1] = np.log1p(np.maximum(x[1], 0)/precip_scale)
    return x


def inverse_v2(x, precip_scale):
    x = np.asarray(x, dtype='float32').copy()
    # Raise rather than silently saturate extreme generated values.
    if not np.isfinite(x).all() or np.max(x[1]) > 30:
        raise FloatingPointError('Nonfinite or overflowing v2 decoded field')
    x[1] = np.expm1(np.maximum(x[1], 0))*precip_scale
    return x
