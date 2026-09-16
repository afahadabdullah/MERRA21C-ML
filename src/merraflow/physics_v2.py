"""Five-channel v2 transforms. No precipitation conservation projection."""
import numpy as np

TARGETS_V2 = ('t2m', 'precip', 'ps', 'u10m', 'v10m')
UNITS_V2 = ('K', 'mm h-1', 'Pa', 'm s-1', 'm s-1')


def precipitation_representation_v2(cfg):
    return cfg.get('representation', {}).get('precip', 'log1p')


def sqrt_precip_v2(rain, scale):
    """sqrt(1 + P/s) - 1, evaluated without cancellation for light rain."""
    value = np.maximum(rain, 0)/scale
    return value/(np.sqrt(1+value)+1)


def encode_fields_v2(x, scale, representation='log1p'):
    if representation == 'log1p':
        return transform_v2(x, scale)
    if representation != 'sqrt1p':
        raise ValueError('Unknown precipitation representation')
    out = np.asarray(x, dtype='float32').copy()
    out[1] = sqrt_precip_v2(out[1], scale)
    return out


def decode_fields_v2(x, scale, representation='log1p'):
    if representation == 'log1p':
        return inverse_v2(x, scale)
    if representation != 'sqrt1p':
        raise ValueError('Unknown precipitation representation')
    out = np.asarray(x, dtype='float32').copy()
    rain = np.maximum(out[1], 0)
    out[1] = scale*rain*(rain+2)
    if not np.isfinite(out).all():
        raise FloatingPointError('Nonfinite square-root decoded field')
    return out


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
