# WMPP ICLR 2027 draft

`main.tex` is the paper; `references.bib` the bibliography; `figures/` and
`tables/` hold the assets. The project compiles standalone (`latexmk -pdf main.tex`)
and automatically switches to the conference style when `iclr2027_conference.sty`
is added to the project root.

## Generated content — do not edit by hand

Everything under `tables/` and the figures `gain_ci.pdf`, `regimes.pdf`,
`kc_profiles.pdf` are produced by

```
python scripts/make_paper_assets.py   # from the repo root, venv activated
```

from the final evaluation report (`/scratch/jwquan/wmpp/planner_eval/og50_final_sweep.json`)
and the env registry (`manifests/wmpp_env_config.json`). `tables/numbers.tex`
defines the macros (`\MeanWMPP`, `\NumSigUp`, `\GainScenePlay`, …) that the running
text uses, so a re-run of the script propagates new numbers everywhere. See
`../RUNBOOK.md` §3.4–3.5 for the full chain (report → assets → compile).

`figures/wmppteaser_optimized.pdf` is the hand-made overview figure.
`figures/current_gain.{pdf,png}` and `figures/gain_ci.pdf` (per-dataset gain bars with CIs) are still generated but no longer referenced by `main.tex`.

## Status (2026-08-25)

19 of 20 datasets are final under the official OGBench protocol; `puzzle-4x6-noisy`
is pending (its policy bank is still queued). Since 2026-08-25 every fixed-policy
baseline is the paired `og50fx` re-evaluation on the planners' episode seeds
(`report_og50.py --fixed_tag=og50fx`), so all deltas are paired per-episode contrasts. Remaining `\TODO` markers in
`main.tex`: qualitative timelines (App. I) and the planned bank-composition /
hysteresis ablations (App. J), both of which need new runs.

## Overleaf

Zip the folder (excluding build artifacts) and upload; set `main.tex` as the main file:

```
cd .. && zip -r WMPP_ICLR2027_$(date +%Y%m%d).zip WMPP_ICLR2027 \
  -x '*.aux' '*.log' '*.out' '*.bbl' '*.blg' '*.fls' '*.fdb_latexmk' '*.synctex.gz'
```
