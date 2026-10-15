# MERRA21C-ML v4.1 manuscript

`MERRA21C_ML_arxiv.tex` is a standalone arXiv-style manuscript with embedded references. It follows the supplied SURMA-Flow example's colored title rule, full-width abstract, Palatino body type, two-column article layout, numbered citations, and running header. It does not reuse that paper's scientific claims, authorship, or results.

The paper covers the latest v4.1 model, as requested. Figure 1 contains the actual GEOS-based workflow and model predictions; four remaining figure interiors and four table interiors are intentionally blank. Their captions specify the intended content. All sections from abstract through conclusions, declarations, appendices, and references are written. The PDF is exported to `../output/pdf/MERRA21C_ML_arxiv.pdf`.

## Build

The arXiv build uses `figures/MERRA21C_ML_workflow.pdf` for Figure 1. The source also embeds a lossless preview fallback because the desktop editor compiler reads only one source file. No bibliography database, private document class, shell escape, or network resources are needed. Standard TeX packages are declared in the preamble.

```sh
pdflatex MERRA21C_ML_arxiv.tex
pdflatex MERRA21C_ML_arxiv.tex
```

Alternatively, use Tectonic or the Codex LaTeX editor. The editor compiler successfully compiled the source during preparation. The local PDF was exported using the bundled Tectonic executable with cached resources.

## Evidence boundary

This is a complete manuscript draft, not a submission-ready claim of established broad skill. The accessible numerical record contains only three selected test hours with EMA weights at epoch 243, eight members, 64 Heun steps, and spectral post-processing. No full test-set skill, confidence interval, observational validation, raw-flow ablation, hardware speedup, or significance result has been invented. All reported case scores come from `present_last_v4_1/metrics.json`; results are not recomputed from unavailable source fields or checkpoints.

The current editor source contains the supplied author names and affiliations. Contributions, competing interests, funding, and public archive identifiers still require author confirmation. The current revision preserves the author block already present in the editor.

## Claim-to-source map

- Model targets, hybrid encoding/decoding, humidity clipping, frozen regression: `src/merraflow/v4.py`.
- Architecture, context attention, Heun patch integrator: `src/merraflow/model_v2.py`.
- Full-domain shared state, Hann velocity assembly, decoding: `src/merraflow/inference_v4.py`.
- Base loss and optional fine-tuning distinction: `src/merraflow/v4_1.py`, `src/merraflow/loss_v2.py`.
- Residual RMS calibration: `src/merraflow/train_v4.py`.
- Packed training implementation: `src/merraflow/packed_v4_1.py`, `src/merraflow/v4_1.py`, `docs/v4_1.md`.
- Current intended settings: `configs/discover_v4_1.yaml`.
- Intended chronological split and inherited regression settings: `configs/discover_annual_v2.yaml`.
- Temporal support, previous-hour eligibility, direct-rain encoding: `src/merraflow/dataset_v3_precip.py`, `docs/v4.md`.
- Static surface and temporal inputs: `src/merraflow/prepare_v2.py`, `src/merraflow/static_v2.py`, `src/merraflow/dataset_v2.py`.
- Grid/experiment identification: `data.md`. Documentation mixes nominal 2 km and 3 km labels; the paper therefore uses kilometer-scale and requires actual geometry for distances.
- Saved numerical evidence: `present_last_v4_1/metrics.json`, `present_last_v4_1/scorecard.csv`, `present_last_v4_1/scorecard_cases.csv`.
- Score aggregation and post-processing invocation: `src/merraflow/present_v4_1.py`.
- Spectral processing and climatology selection: `src/merraflow/explore_inference_v4_1.py`.
- CRPS, population-normalized spread, empirical quantile coverage, FSS, rank ties, radial spectra: `src/merraflow/metrics.py`.

## Items needed before submission

1. Confirm authorship and all declarations; replace the draft-status label and author placeholders only after author approval.
2. Retrieve production checkpoint metadata, hashes, the frozen regression training record, retained manifests, normalization arrays, and target-cache fingerprints. The checked-in configuration establishes intended settings, not proof of every production-run setting.
3. Confirm actual grid distances, source collection versions, and data permissions. GEOS-FP distributed output resolution is different from its underlying model integration resolution. GEOS-FP must not be identified as MERRA-2.
4. Confirm no training/validation/test leakage through the frozen regression, normalization, target windows, or spectral climatology. The presentation pipeline requests 12 training reference hours near the first case and reuses them across cases, but the stored summary omits their dates. Its climatology helper can fall back to same-split hours if no training pool exists. A publication run should require training-only selection and record all source dates.
5. Evaluate a validation-selected checkpoint on untouched full-domain episodes. The existing record is the latest checkpoint at epoch 243, not documented as the validation-best checkpoint. Disclose if test cases were used to choose inference recipes.
6. Compare raw flow and spectral processing using identical members, plus matched deterministic and generative baselines. Complete the representation/history/context/proposal/tiling ablations before claiming their incremental benefit.
7. Add episode-block uncertainty intervals, threshold sample sizes, event reliability, spatial skill, multivariate checks, and additional held-out seasons. Three selected hours cannot establish population-wide significance.
8. Preserve the distinction between empirical CRPS and fair CRPS. The saved spread is population-normalized, and the scorecard multiplies spread/RMSE by sqrt((M+1)/M); Appendix B explains why this is not the full finite-member variance correction for that spread definition. Do not silently mix conventions when replacing the results.
9. Quantify production compute cost separately from plotting and output. The recorded 3822 s is the presentation run's total time, not pure sampling time.
10. Fill the remaining four figures and four tables with verified results, update the abstract and conclusions to the final evidence, and add persistent public code/data/checkpoint identifiers where release is permitted.

## References

Fourteen references are embedded in the TeX source. Titles and identifiers were checked against primary publisher, NASA, and arXiv records on 14 October 2026. arXiv references are identified as such. The current 2026 GEOS-FP specification is cited for product definitions, not used to assert that all 2024–2026 source files use the latest operational configuration.

## CONUS revision and full-test evaluation

The current source adds the CorrDiff comparison, the reported qualitative precipitation and humidity findings, and the implemented complete-test protocol. See `../docs/evaluation_all_v4_1.md`. Global extension is mentioned only as future work. No production full-test results have been inserted. The existing PDF and source ZIP are exports of the initial draft; this revision updates the open TeX source and its editor preview only, as requested.

## GEOS workflow figure and actual predictions

Figure 1 is now populated at the author's request. It uses the original `20260307_1930` 2 m temperature panels from `present_last_v4_1`, with native GEOS-FP input, HWT GEOS replay reference, a generated member and the eight-member mean, all on the original common temperature scale in degrees Celsius. These are actual epoch-243 EMA, 64-step, spectral v4.1 outputs. The reference is dynamical GEOS replay; the model predictions are learned downscaling. Nominal 2–3 km labels follow the archive ambiguity already documented in the data section. No HRRR maps are used.

Reproduce the two figure layouts with `python -m merraflow.workflow_archive_v4_1`. The compact vector PDF is included in `figures/MERRA21C_ML_workflow.pdf`; the broad layout appears in the repository README. `figures/workflow_provenance.json` records the source map hashes and clips. The compositor reuses rendered source plots without inferring values from images or rerunning evaluation. These qualitative panels do not add full-test statistics.

To update the editor fallback after regenerating the figure, run:

```bash
python scripts/embed_workflow_preview.py --tex manuscript/MERRA21C_ML_arxiv.tex \
  --pdf manuscript/figures/MERRA21C_ML_workflow.pdf
```

The normal build selects the external vector PDF; the single-source editor uses a 144-dpi PNG preview, embedded with its RGB pixels losslessly preserved. The built-in compiler successfully compiled the updated source with Figure 1. Existing initial-draft PDF/ZIP exports remain stale; the open editor source is the current manuscript.
