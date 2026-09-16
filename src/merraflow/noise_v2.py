"""Consistent initial noise for tiled inference, including outside-domain halos."""
import numpy as np


NOISE_PADDING_MODES = ('independent_halo', 'replicate')


def noise_padding_v2(cfg):
    mode = cfg['inference'].get('noise_padding', 'independent_halo')
    if mode not in NOISE_PADDING_MODES:
        raise ValueError(f'Invalid v2 noise_padding: {mode}')
    return mode


def saved_noise_padding_v2(attrs):
    # Files predating the fix used replicated noise and did not record a mode.
    mode = attrs.get('noise_padding', 'replicate')
    if mode not in NOISE_PADDING_MODES:
        raise ValueError(f'Invalid saved noise_padding: {mode}')
    return mode


def padded_noise_v2(seed, shape, halo, mode='independent_halo'):
    """Preserve every legacy in-domain draw; fill a shared independent halo.

    Coordinates in this extended array are shared by all overlapping tiles.
    Generating a fresh full padded field first would change the in-domain draws
    and confound an old-versus-new comparison, so draw the original field first.
    """
    if mode not in NOISE_PADDING_MODES or halo < 0:
        raise ValueError('Invalid noise padding mode or halo')
    rng = np.random.default_rng(seed)
    domain = rng.standard_normal(shape, dtype=np.float32)
    if halo == 0:
        return domain
    if mode == 'replicate':
        return np.pad(domain, ((0, 0), (halo, halo), (halo, halo)), mode='edge')
    channels, h, w = shape
    padded = np.empty((channels, h+2*halo, w+2*halo), dtype=np.float32)
    padded[:, halo:halo+h, halo:halo+w] = domain
    # Disjoint strips, including corners, each receive independent draws.
    for region in (np.s_[:, :halo, :], np.s_[:, halo+h:, :],
                   np.s_[:, halo:halo+h, :halo], np.s_[:, halo:halo+h, halo+w:]):
        padded[region] = rng.standard_normal(padded[region].shape, dtype=np.float32)
    return padded
