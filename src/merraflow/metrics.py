"""Area-weighted physical-unit deterministic and finite-ensemble verification."""
import numpy as np
from scipy.ndimage import uniform_filter


def weighted_mean(x, area):
    return float(np.sum(np.asarray(x, dtype='float64')*area)/np.sum(area))


def crps_ensemble(ensemble, truth):
    """Empirical CRPS in O(M log M), without an M-by-M pairwise allocation."""
    m = len(ensemble)
    sorted_x = np.sort(ensemble.astype('float64'), axis=0)
    coeff = (2*np.arange(1, m+1)-m-1).reshape((m,)+(1,)*(ensemble.ndim-1))
    return np.mean(np.abs(ensemble-truth), axis=0)-np.sum(coeff*sorted_x, axis=0)/(m*m)


def continuous(ensemble, truth, area):
    pred = ensemble.mean(0)
    diff = pred-truth
    pm, tm = weighted_mean(pred, area), weighted_mean(truth, area)
    covariance = weighted_mean((pred-pm)*(truth-tm), area)
    denominator = np.sqrt(weighted_mean((pred-pm)**2, area)*weighted_mean((truth-tm)**2, area))
    lo, hi = np.quantile(ensemble, [.05, .95], axis=0)
    return {'rmse': np.sqrt(weighted_mean(diff**2, area)), 'mae': weighted_mean(abs(diff), area),
            'bias': weighted_mean(diff, area), 'correlation': covariance/denominator if denominator else None,
            'crps': weighted_mean(crps_ensemble(ensemble, truth), area),
            'spread': np.sqrt(weighted_mean(np.var(ensemble, axis=0), area)),
            'coverage_90': weighted_mean((truth >= lo) & (truth <= hi), area)}


def fss(pred, truth, threshold, scale, area):
    """Neighborhood event fractions; score only centers with full in-domain windows."""
    if scale % 2 != 1 or scale < 1:
        raise ValueError('FSS scale must be an odd positive pixel width')
    if scale > min(pred.shape):
        return None
    p = uniform_filter((pred >= threshold).astype('float64'), size=scale, mode='constant')
    t = uniform_filter((truth >= threshold).astype('float64'), size=scale, mode='constant')
    radius = scale//2
    region = (slice(radius, -radius), slice(radius, -radius)) if radius else (slice(None), slice(None))
    p, t, a = p[region], t[region], area[region]
    denominator = weighted_mean(p*p+t*t, a)
    return 1-weighted_mean((p-t)**2, a)/denominator if denominator > 0 else None


def precipitation(ensemble, truth, area, thresholds=(.1, 1, 5, 10, 25), scales=(1, 5, 17, 33)):
    pred = ensemble.mean(0)
    result = {}
    for threshold in thresholds:
        p, t = pred >= threshold, truth >= threshold
        hit, miss, false = [weighted_mean(v, area) for v in (p & t, ~p & t, p & ~t)]
        probability = (ensemble >= threshold).mean(0)
        bins = []
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            mask = (probability >= lo) & ((probability <= hi) if hi == 1 else (probability < hi))
            mass = area[mask].sum()
            bins.append({'lower': float(lo), 'upper': float(hi), 'count': int(mask.sum()), 'area_m2': float(mass),
                         'forecast_probability': weighted_mean(probability[mask], area[mask]) if mass else None,
                         'observed_frequency': weighted_mean(t[mask], area[mask]) if mass else None})
        result[str(threshold)] = {'csi': hit/(hit+miss+false) if hit+miss+false else None,
                                 'pod': hit/(hit+miss) if hit+miss else None,
                                 'far': false/(hit+false) if hit+false else None,
                                 'frequency_bias': weighted_mean(p, area)/weighted_mean(t, area) if t.any() else None,
                                 'brier': weighted_mean((probability-t)**2, area), 'reliability': bins,
                                 'ensemble_mean_fss': {str(s): fss(pred, truth, threshold, s, area) for s in scales},
                                 'member_fss_mean': {str(s): _mean_defined([fss(member, truth, threshold, s, area) for member in ensemble]) for s in scales}}
    result['pixel_quantiles'] = {'probabilities': [.5, .95, .99, .999],
                                 'truth': np.quantile(truth, [.5, .95, .99, .999]).tolist(),
                                 'members_pooled': np.quantile(ensemble, [.5, .95, .99, .999]).tolist()}
    return result


def _mean_defined(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def rank_histogram(ensemble, truth, area, seed=0):
    """Randomize ties (especially dry precipitation) instead of biasing rank zero."""
    rng = np.random.default_rng(seed)
    lower, ties = (ensemble < truth).sum(0), (ensemble == truth).sum(0)
    rank = lower+np.floor(rng.random(truth.shape)*(ties+1)).astype(int)
    hist = np.bincount(rank.ravel(), weights=area.ravel(), minlength=len(ensemble)+1)
    return (hist/hist.sum()).tolist()


def radial_psd(field):
    """Detrended Hann-tapered PSD vs cycles/grid pixel (LCC index-space diagnostic)."""
    h, w = field.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    f = (field-field.mean())*window
    power = abs(np.fft.rfft2(f))**2/(h*w*np.mean(window**2))
    fy, fx = np.fft.fftfreq(h), np.fft.rfftfreq(w)
    radius = np.sqrt(fy[:, None]**2+fx[None, :]**2)
    edges = np.linspace(0, .5, min(h, w)//2+1)
    index = np.digitize(radius.ravel(), edges)-1
    valid = (index >= 0) & (index < len(edges)-1)
    count = np.bincount(index[valid], minlength=len(edges)-1)
    summed = np.bincount(index[valid], weights=power.ravel()[valid], minlength=len(edges)-1)
    psd = np.divide(summed, count, out=np.zeros_like(summed), where=count > 0)
    return (edges[:-1]+edges[1:])/2, psd
