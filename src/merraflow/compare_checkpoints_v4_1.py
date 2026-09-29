"""Compare checkpoints on one case with the same sampling recipe(s), same member noise and region.

Each checkpoint runs through ``explore_inference_v4_1`` (baseline and, by default, the
``spectral`` recipe) into its own sub-folder, then the scores are gathered into one table:
CRPS, ensemble-mean RMSE and MAE, bias, spread/skill, p99 |grad| ratio, 4-30 km power ratio
and the edge step ratio at the truth's sharpest edges, per field.

Ranking: mean over fields of CRPS relative to the best checkpoint for that field (1 = best in
every field). One case with 8 members is a screen; confirm on the test set.

Runs are given as ``label=config:checkpoint`` (checkpoint: best | latest | <epoch> | <path>):

  main_best=configs/discover_v4_1.yaml:best
  main_latest=configs/discover_v4_1.yaml:latest
  ft1_best=configs/discover_v4_1_ft_rollout.yaml:best
  ft2_best=configs/discover_v4_1_ft_rollout2.yaml:best

Outputs (default <first run's train.output>/evaluation/checkpoints_<case>_<job>/):
  report.md, summary.json, checkpoints.{png,pdf}; <label>/ (full explore outputs per checkpoint)
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from .config import write_json
from .v4 import TARGETS
from .v4_1 import load_config
from .explore_inference_v4_1 import run as explore, OPTION_DEFAULTS, EDGE_FIELDS

DEFAULT_RUNS = ('main_best=configs/discover_v4_1.yaml:best',
                'main_latest=configs/discover_v4_1.yaml:latest',
                'ft1_best=configs/discover_v4_1_ft_rollout.yaml:best',
                'ft2_best=configs/discover_v4_1_ft_rollout2.yaml:best')
METRICS = (('crps', 'CRPS', '{:.4g}'), ('rmse', 'Ensemble-mean RMSE', '{:.4g}'),
           ('mae', 'Ensemble-mean MAE', '{:.4g}'), ('bias', 'Bias (mean − truth)', '{:+.3g}'),
           ('spread_skill', 'Spread/skill (1 = calibrated)', '{:.2f}'),
           ('p99_ratio', 'p99 |∇| ratio to truth', '{:.2f}'), ('fine_psd_ratio', '4–30 km power ratio', '{:.2f}'))


def parse_runs(items):
    runs = []
    for item in items:
        label, _, rest = item.partition('=')
        config, _, checkpoint = rest.rpartition(':')
        if not (label and config and checkpoint):
            raise ValueError(f'Run must be label=config:checkpoint, got {item!r}')
        runs.append((label.strip(), config.strip(), checkpoint.strip()))
    if len({r[0] for r in runs}) != len(runs):
        raise ValueError('Run labels must be unique')
    return runs


def run(runs, timestamp=None, split='val', recipes=('baseline', 'spectral'), center=None, center_latlon=None,
        weights='ema', output=None, batch=32, threads=8, dpi=200, pdf=True, log=print, configs=None, **options):
    """runs: [(label, config path, checkpoint)]; ``configs`` optionally maps config path -> loaded
    config (tests). Returns the output folder."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    started = time.monotonic()
    load = (lambda path: configs[path]) if configs else load_config
    job = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
    first = load(runs[0][1])
    out = Path(output) if output else (Path(first['train']['output'])/'evaluation'/
                                      f'checkpoints_{timestamp or "wettest"}_{job}')
    out.mkdir(parents=True, exist_ok=True)
    o = dict(dict(members=8, pool=8), **options, phase2=False, combine=())
    results = {}
    for label, config, checkpoint in runs:
        log(f'=== {label}: {config} checkpoint {checkpoint} ===')
        folder = explore(load(config), checkpoint, timestamp, split, tuple(recipes), center, center_latlon, False,
                         weights, out/label, batch, threads, dpi, pdf, False, log=log, **o)
        results[label] = json.loads((folder/'metrics.json').read_text())
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(results, recipes)
    write_json(out/'summary.json', json.loads(json.dumps(summary, default=float)))
    fig = plot(summary, recipes, plt)
    fig.savefig(out/'checkpoints.png', dpi=dpi)
    if pdf:
        fig.savefig(out/'checkpoints.pdf', dpi=dpi)
    plt.close(fig)
    write_report(out, summary, recipes)
    log(f'Best (mean relative CRPS): {summary["ranking"][0]} · written to {out} ({time.monotonic()-started:.0f}s)')
    return out


def summarize(results, recipes):
    rows = {}
    for label, m in results.items():
        for recipe in recipes:
            if recipe not in m['results']:
                continue
            fields = m['results'][recipe]['fields']
            rows[f'{label} · {recipe}'] = dict(
                label=label, recipe=recipe, epoch=m['epoch'], checkpoint=m['checkpoint'],
                gap=m['results'][recipe]['gap'], fields=fields)
    fields = list(TARGETS)
    best = {f: min(r['fields'][f]['crps'] for r in rows.values()) for f in fields}
    for r in rows.values():
        r['relative_crps'] = {f: r['fields'][f]['crps']/max(best[f], 1e-30) for f in fields}
        r['mean_relative_crps'] = float(np.mean(list(r['relative_crps'].values())))
        edges = [r['fields'][f].get('edge_ratio') for f in ('q2m', 't2m')]
        r['edge_q2m_t2m'] = float(np.mean(edges)) if all(e is not None for e in edges) else None
    ranking = sorted(rows, key=lambda k: rows[k]['mean_relative_crps'])
    case = next(iter(results.values()))
    return dict(case=case['case'], time=case['time'], region=case['region'], members=case['options']['members'],
                rows=rows, ranking=ranking, fields=fields)


def plot(summary, recipes, plt):
    rows, fields = summary['rows'], summary['fields']
    names = summary['ranking']
    fig, axes = plt.subplots(1, 3, figsize=(21, max(4.5, .5*len(names)+2)), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    data = np.array([[rows[n]['relative_crps'][f] for f in fields] for n in names])
    image = axes[0].imshow(data, cmap='Reds', vmin=1, vmax=max(1.05, float(data.max())), aspect='auto')
    for (i, j), v in np.ndenumerate(data):
        axes[0].text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8)
    axes[0].set_xticks(range(len(fields)), fields)
    axes[0].set_yticks(range(len(names)), names, fontsize=8)
    axes[0].set_title('CRPS relative to the best checkpoint per field (1 = best)', fontsize=10)
    fig.colorbar(image, ax=axes[0], shrink=.7)
    y = np.arange(len(names))[::-1]
    axes[1].barh(y, [rows[n]['mean_relative_crps'] for n in names], color='#2166ac')
    axes[1].set_yticks(y, names, fontsize=8)
    axes[1].set_xlim(.99*min(rows[n]['mean_relative_crps'] for n in names),
                     1.01*max(rows[n]['mean_relative_crps'] for n in names))
    axes[1].set_title('Mean relative CRPS (lower = better)', fontsize=10)
    width = .8/len(EDGE_FIELDS)
    for k, f in enumerate(EDGE_FIELDS):
        axes[2].barh(y+(k-(len(EDGE_FIELDS)-1)/2)*width, [rows[n]['fields'][f].get('edge_ratio') or 0 for n in names],
                     width, label=f)
    axes[2].axvline(1, color='k', lw=1)
    axes[2].set_yticks(y, names, fontsize=8)
    axes[2].legend(fontsize=8)
    axes[2].set_title('Edge step ratio at truth edges (1 = truth)', fontsize=10)
    fig.suptitle(f'Checkpoint comparison · case {summary["case"]} · {summary["members"]} members · '
                 f'recipes: {", ".join(recipes)}', fontsize=12)
    return fig


def _f(v, fmt):
    return 'n/a' if v is None else fmt.format(v)


def write_report(out, summary, recipes):
    rows, fields = summary['rows'], summary['fields']
    L = [f'# Checkpoint comparison · case `{summary["case"]}`', '',
         f'Same case, region {summary["region"]}, member noise and recipes ({", ".join(recipes)}); '
         f'{summary["members"]} members. One case: a screen, not a verdict.', '',
         f'**Best by mean relative CRPS: `{summary["ranking"][0]}`**', '',
         '## Ranking', '',
         '| Rank | Checkpoint · recipe | Epoch | Mean relative CRPS | Edge q2m | Edge t2m | Sharpness gap | '
         + ' | '.join(f'CRPS {f}' for f in fields) + ' |',
         '|---:|---|---:|---:|---:|---:|---:|' + '---:|'*len(fields)]
    for i, n in enumerate(summary['ranking'], 1):
        r = rows[n]
        L.append(f'| {i} | `{n}` | {r["epoch"]} | {r["mean_relative_crps"]:.4f} | '
                 f'{_f(r["fields"]["q2m"].get("edge_ratio"), "{:.2f}")} | '
                 f'{_f(r["fields"]["t2m"].get("edge_ratio"), "{:.2f}")} | {r["gap"]:.3f} | '
                 + ' | '.join(f'{r["fields"][f]["crps"]:.4g}' for f in fields) + ' |')
    for key, title, fmt in METRICS[1:]:
        L += ['', f'## {title}', '', '| Checkpoint · recipe | ' + ' | '.join(fields) + ' |', '|---|' + '---:|'*len(fields)]
        for n in summary['ranking']:
            L.append(f'| `{n}` | ' + ' | '.join(_f(rows[n]['fields'][f].get(key), fmt) for f in fields) + ' |')
    L += ['', '## Edge step ratio at the truth\'s sharpest edges (1 = as sharp as truth)', '',
          '| Checkpoint · recipe | ' + ' | '.join(EDGE_FIELDS) + ' |', '|---|' + '---:|'*len(EDGE_FIELDS)]
    for n in summary['ranking']:
        L.append(f'| `{n}` | ' + ' | '.join(_f(rows[n]['fields'][f].get('edge_ratio'), '{:.2f}')
                                            for f in EDGE_FIELDS) + ' |')
    L += ['', 'Checkpoints: ' + '; '.join(f'`{r["label"]}` = {r["checkpoint"]}' for r in
                                           {rows[n]['label']: rows[n] for n in rows}.values()),
          '', 'Per-checkpoint maps, edge zooms and spectra are in the sub-folders.']
    (out/'report.md').write_text('\n'.join(L)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', default=list(DEFAULT_RUNS), help='label=config:checkpoint ...')
    parser.add_argument('--recipes', default='baseline,spectral', help='Explore recipes (default baseline,spectral)')
    parser.add_argument('--timestamp')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--center', type=int, nargs=2, metavar=('ROW', 'COL'))
    parser.add_argument('--center-latlon', type=float, nargs=2, metavar=('LAT', 'LON'))
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--members', type=int, default=8)
    parser.add_argument('--steps', type=int, default=OPTION_DEFAULTS['steps'])
    parser.add_argument('--region', type=int, default=OPTION_DEFAULTS['region'])
    parser.add_argument('--spectral-max-gain', type=float, default=1.3)
    parser.add_argument('--output')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--no-pdf', action='store_true')
    a = parser.parse_args()
    run(parse_runs(a.runs), a.timestamp, a.split, tuple(r for r in a.recipes.split(',') if r),
        tuple(a.center) if a.center else None, tuple(a.center_latlon) if a.center_latlon else None, a.weights,
        a.output, a.batch, a.threads, a.dpi, not a.no_pdf, log=lambda message: print(message, flush=True),
        members=a.members, pool=a.members, steps=a.steps, region=a.region, spectral_max_gain=a.spectral_max_gain)


if __name__ == '__main__':
    main()
