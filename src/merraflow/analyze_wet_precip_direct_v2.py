"""CPU-only spatial audit of an existing direct-precipitation wet evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import fss


THRESHOLDS = (1., 5., 10.)
SCALES = (1, 5, 17, 33)
FIELDS = ('member_mean', 'ensemble_mean', 'coarse', 'regression')


def spatial_scores(ensemble, truth, coarse, regression, area,
                   thresholds=THRESHOLDS, scales=SCALES):
    """FSS of individual members and deterministic fields on the same patch."""
    ensemble, truth, coarse, area = map(np.asarray, (ensemble, truth, coarse, area))
    if ensemble.ndim != 3 or ensemble.shape[1:] != truth.shape or any(
            field.shape != truth.shape for field in (coarse, area)):
        raise ValueError('Expected members and matching two-dimensional patch fields')
    if regression is not None and np.asarray(regression).shape != truth.shape:
        raise ValueError('Regression field must match the truth patch')
    if not all(np.isfinite(field).all() for field in (ensemble, truth, coarse, area)):
        raise ValueError('Nonfinite spatial-audit field')
    if np.any(area <= 0):
        raise ValueError('Area weights must be positive')
    fields = dict(ensemble_mean=ensemble.mean(0), coarse=coarse)
    if regression is not None:
        fields['regression'] = np.asarray(regression)
    result = {}
    for threshold in thresholds:
        by_scale = {}
        for scale in scales:
            member_scores = [fss(member, truth, threshold, scale, area) for member in ensemble]
            scores = {name: fss(field, truth, threshold, scale, area)
                      for name, field in fields.items()}
            defined = [score for score in member_scores if score is not None]
            scores['member_mean'] = float(np.mean(defined)) if defined else None
            by_scale[str(scale)] = scores
        result[str(threshold)] = by_scale
    return result


def uniform_epoch(history_path, epoch):
    history = json.loads(Path(history_path).read_text())
    matches = [row for row in history if row.get('epoch') == epoch and 'crps' in row]
    if len(matches) != 1:
        raise ValueError(f'Expected one uniform validation row for epoch {epoch} in {history_path}')
    row = matches[0]
    keys = ('epoch', 'crps', 'coarse_crps', 'regression_crps', 'bias', 'spread',
            'ensemble_mean_rmse', 'wet_fraction', 'truth_wet_fraction')
    result = {key: row[key] for key in keys if key in row}
    if row.get('coarse_crps'):
        result['crps_improvement_vs_coarse_percent'] = 100*(1-row['crps']/row['coarse_crps'])
    if row.get('regression_crps'):
        result['crps_improvement_vs_regression_percent'] = 100*(1-row['crps']/row['regression_crps'])
    return result


def analyze(results, history_path=None, output=None):
    results = Path(results)
    report = json.loads((results/'report.json').read_text())
    cases = report['cases']
    if not cases:
        raise ValueError('No wet evaluation cases')
    case_scores = []
    for index, case in enumerate(cases, 1):
        path = results/f'case_{index:02d}_{case["id"]}.npz'
        with np.load(path, allow_pickle=False) as arrays:
            regression = arrays['regression']
            scores = spatial_scores(arrays['ensemble'], arrays['truth'], arrays['coarse'],
                                    regression if regression.size else None, arrays['area'])
        case_scores.append(dict(id=case['id'], time=case['time'], fss=scores))
    aggregated = {}
    for threshold in THRESHOLDS:
        key = str(threshold)
        aggregated[key] = {}
        for scale in SCALES:
            scale_key = str(scale)
            aggregated[key][scale_key] = {}
            for name in FIELDS:
                values = [item['fss'][key][scale_key].get(name) for item in case_scores]
                defined = [value for value in values if value is not None]
                aggregated[key][scale_key][name] = (float(np.mean(defined)) if defined else None)
    audit = dict(epoch=report['epoch'], cases=len(cases),
                 scales_pixels=list(SCALES), nominal_km_per_pixel=3,
                 thresholds_mm_h=list(THRESHOLDS),
                 note='FSS=1 is perfect; null means no event at that threshold and scale. '
                      'Member mean averages FSS of individual members, not their rainfall fields. '
                      'Only selected wet patches are scored; uniform validation is separate.',
                 mean_wet_case_fss=aggregated, cases_detail=case_scores,
                 uniform_validation=(uniform_epoch(history_path, report['epoch'])
                                     if history_path is not None else None))
    destination = Path(output) if output is not None else results/'spatial_report.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2)+'\n')
    plot_scores(audit, destination.with_name('spatial_skill.png'))
    return destination


def plot_scores(audit, path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(14, 4),
                             constrained_layout=True, sharey=True)
    labels = dict(member_mean='Mean member', ensemble_mean='Ensemble mean',
                  coarse='Coarse', regression='Frozen v2')
    widths = np.asarray(SCALES)*audit['nominal_km_per_pixel']
    for ax, threshold in zip(axes, THRESHOLDS):
        data = audit['mean_wet_case_fss'][str(threshold)]
        for name in FIELDS:
            values = [data[str(scale)][name] for scale in SCALES]
            if any(value is not None for value in values):
                ax.plot(widths, [np.nan if value is None else value for value in values],
                        marker='o', label=labels[name])
        ax.set(title=f'{threshold:g} mm/h event', xlabel='Neighborhood width (nominal km)',
               ylim=(0, 1), xticks=widths)
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Fraction skill score')
    axes[-1].legend(loc='lower right')
    fig.suptitle(f'Epoch {audit["epoch"]}: spatial skill on {audit["cases"]} wet validation cases')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, help='Wet evaluation results folder')
    parser.add_argument('--history', help='Training history.json with uniform validation rows')
    parser.add_argument('--output', help='Output JSON; default: results/spatial_report.json')
    args = parser.parse_args()
    print(analyze(args.results, args.history, args.output))


if __name__ == '__main__':
    main()
