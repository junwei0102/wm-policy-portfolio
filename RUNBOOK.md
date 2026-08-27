# WMPP RUNBOOK — World-Model Policy Portfolios on OGBench

Operational handbook for anyone (human or agent) working in this repo. It records
the state of the project on **2026-08-25**, the exact protocol behind every
number in the paper, how to reproduce or extend each stage, and the rules that
keep parallel agents from stepping on each other.

Read this file completely before submitting a job or editing the paper.

---

## 0. Ground truth in one screen

| Item | Value |
|---|---|
| Repo | `/project/6067317/jwquan/wm-policy-portfolio` (this directory; **git has no commits yet** — see §9) |
| Paper | `WMPP_ICLR2027/main.tex` (Overleaf export; tables/figures auto-generated into `WMPP_ICLR2027/{tables,figures}/`) |
| OGBench checkout | `/project/6067317/jwquan/ogbench` (v1.2.1, commit `1d41409`, never modified; `impls/` is imported via `OGBENCH_IMPLS`) |
| Python env | `source /scratch/jwquan/wmpp/venv/bin/activate` (Python 3.10, JAX CPU/GPU, mujoco 3.1.6, matplotlib) |
| Env vars for evals | `export OGBENCH_IMPLS=/project/6067317/jwquan/ogbench/impls MUJOCO_GL=disable JAX_PLATFORMS=cpu` |
| Scratch root | `/scratch/jwquan/wmpp/` → `policies/`, `wm/`, `planner_eval/`, `oracle/`, `launchers/`, `logs/`, `data/` |
| Final result table (19 envs) | `/scratch/jwquan/wmpp/planner_eval/og50_final_sweep.{md,json}` |
| Env registry | `manifests/wmpp_env_config.json` (per env: policy root, WM dir/epoch, eval.csv baselines per seed, best family) |
| Headline | mean success over 19 datasets: **WMPP 42.2 / best fixed policy 31.9 / Random-Switch 27.3** (best policy = paired `og50fx` re-evaluation, see §1); ΔWMPP +10.3, **8 sig↑ / 0 sig↓ / 11 n.s.** (paired CIs); WMPP−Random sig in 14/19, never reversed |
| Pending | `puzzle-4x6-noisy-v0` (bank array `54662964` rows 91–108: 7/18 done, rest PENDING on fair-share Priority) |
| SLURM | CPU evals: `--account=def-gigor_cpu --cpus-per-task=8 --mem=24G --exclude=fc30554,fc30560,fc30572,fc30657`; GPU training: `--account=def-gigor_gpu --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1` (2g.20gb fine for WMs), `--exclude=fc10507,fc10512,fc11013` |

---

## 1. What the project is

**Claim.** Given a *bank* of frozen goal-conditioned policies trained by different
GCRL algorithms on the same offline dataset, a learned state-space world model
plus a shared metric value function can pick, at test time, which policy should
act *now*, and this beats the best single policy without retraining anything.

**Method (WMPP = `planners/portfolio.py:RolloutRanker`).** Every `c` real env
steps: roll each bank policy forward `k` imagined steps through an ensemble of
`E=3` one-step MLP dynamics models (closed loop: the policy acts on its own
imagined states), score every imagined state with the ensemble-mean LAVL metric
value `V(s,g) = -||φ_S(s) − φ_G(ψ(g))||`, take the **max over the horizon**
(`score_agg=max`, `score_mode=value`), execute the arg-max policy for `c` steps,
repeat. We only use the diagonal `k = c`, so the nominal model budget per env
step is `P·E` regardless of the cell — `(k,c)` is a zero-overhead hyperparameter.

**Ablation (the only one in the paper): `RandomArbiter`** — uniform draw from
the full bank every `c` steps, no model calls, same `c` as the selected WMPP cell.
It isolates model-based *selection* from the *commitment schedule*.

**Bank formalism.** One checkpoint per algorithm per bank seed
`r ∈ {0,1,2}`: `Π_r = {GCBC, GCIVL, GCIQL, QRL, CRL, HIQL}` (HIQL absent in the
six `p2-main` envs → P=5 there). Everything is replicated over the 3 bank seeds.

**Evaluation protocol ("og50", user-locked 2026-08-24).** Official OGBench
protocol: task_ids 1–5 × 50 episodes = 250 episodes per bank seed, deterministic
per-episode reset seeds shared by every method, temperature 0. **Fixed-policy
baselines (user decision 2026-08-25, "option 2"): every bank policy is re-evaluated
on the planners' episode seeds** (tag `og50fx`, `launchers/launch_og50fx.sh`), so
every Δ in the paper — vs best policy and vs Random — is a *paired* per-episode
contrast under the same hierarchical bootstrap. Each run's own `eval.csv`
(recorded in `manifests/wmpp_env_config.json`) is kept only as a cross-check: it
agrees with the paired re-evaluation to within 3.3 pt (mean |diff| 0.7 pt over
108 env×family pairs) and picks the same best family on 18/19 envs (the exception
is the all-zero tie on cube-quadruple-play). The eval.csv-based report is archived
as `planner_eval/og50_final_sweep_evalcsv.{md,json}`.
"Best policy" = algorithm family with the highest 3-seed mean on the paired episodes.

**Selection rule.** Per env, the best diagonal cell `k=c ∈ {1,5,10,25,50,100}`
by 3-seed mean success (ties → smallest k); Random is paired at that `c`.
Selected k: antmaze 5, cube-double-noisy 5, cube-double-play 10,
cube-quadruple-noisy 25, cube-quadruple-play 1 (all-zero tie), cube-single-noisy 5,
cube-single-play 1, cube-triple-noisy 50, cube-triple-play 100, humanoid 50,
puzzle-3x3-noisy 100, 3x3-play 50, 4x4-noisy 25, 4x4-play 100, 4x5-noisy 50,
4x5-play 50, 4x6-play 100, scene-noisy 5, scene-play 1.

**Statistics.** Δ vs best policy: hierarchical bootstrap (resample bank seeds,
then episodes within seed; 10k resamples; 95 % percentile CI) of the method's
750 episodes, offset by the eval.csv baseline. Method-vs-method: paired per
episode, same hierarchical scheme. No multiple-comparison correction (state
this in the paper when counting significant rows).

**Decisions that are locked (do not re-litigate; ask the user to change them):**
- Main number = best diagonal `k=c` cell. `(100,1)` / off-diagonal cells are
  **dropped from the paper** (data remains on disk under `*_h1` tags).
- `(1,1)` is a member of the k=c sweep, not a separate "One-Step" control.
- Random-Switch is the only ablation column. The old "Value-Score" (Q-only)
  baseline was never run and is out of the paper.
- Visual/pixel experiments are out of the paper (assets kept on disk).
- Report absolute success tables (every policy + WMPP), not deltas alone.
- All paper numbers come from the og50 protocol (planners: `og50*`; fixed
  policies: `og50fx`, paired). The older 20-episode runs (`lavlp`, `lavlp_h1`,
  `abl`, `lavlv*`, `hsweep`, …) are superseded and no longer feed the paper
  (`og50fx` also supplies the fixed-policy ms/step anchor and the static oracle).
- Baselines are the paired `og50fx` re-evaluation, not `eval.csv` (2026-08-25).
  Consequence: paired CIs are wider than the old constant-offset ones, so
  antmaze (+2.3), puzzle-4x6-play (+2.3) and the cube-single-play loss (−3.7)
  are **not** significant any more; cube-single-noisy (+0.8) is. 8↑/0↓/11 n.s.

---

## 2. Directory map

```
wm-policy-portfolio/
  interfaces/      FrozenPolicy (restores impls checkpoints, deterministic act), PolicyBank, SimBranch
  world_model/     model.py (EnsembleWorldModel + LANValue head, get_config), seq_dataset.py, rollout.py
  planners/        portfolio.py: OneStepChooser, RolloutRanker (WMPP), RandomArbiter
  evaluation/      paired_eval.py (episode harness), oracle.py, diagnostics.py
  scripts/         train_policy.py, train_world_model.py, eval_planner.py, report_og50.py,
                   report_ablation.py, aggregate_planner_eval.py, make_paper_assets.py,
                   slurm_*.sbatch templates, *_manifest.csv (policy-bank launch rows)
  manifests/       wmpp_env_config.json (env registry), run_manifest.schema.json
  tests/           pytest gates (determinism, planners, WM)
  WMPP_ICLR2027/   paper: main.tex, references.bib, figures/, tables/ (generated)
  RUNBOOK.md       this file

/scratch/jwquan/wmpp/
  policies/{p1,p2,mpilot,visual}/OGBench/<run_group>/<env>/<algo>_sd<s>.../params_1000000.pkl + eval.csv
  wm/wmogbench/wm-value-lavl/sd000_s_<jobid>.<ts>/{flags.json, params_1000000.pkl, train.csv}
  wm/wmogbench/wm-value-lavl-h{50,100}/   horizon-ablation WMs (appendix-only, superseded)
  planner_eval/<env>/bank_sd<s>_<tag>/{episodes.csv, summary.json}
  planner_eval/og50_final_sweep.{md,json}  <- FINAL report
  planner_eval/og50_main_table.tex         <- main table (also regenerated by make_paper_assets.py)
  oracle/<env>/                            dynamic-oracle branches + diagnostics*.json (cube-double-play etc.)
  launchers/launch_og50{,k5,r1,r1_cqp}.sh  exact sbatch lines that produced every og50 run
  logs/<jobname>_<jobid>.out
```

Policy roots per env (from `wmpp_env_config.json`): `p1-pilot` = antmaze-large,
cube-double-play; `p2-main` = cube-quadruple-play, cube-triple-play/noisy,
humanoidmaze-giant, puzzle-4x4-play, scene-play (5 algos, no HIQL);
`mpilot` = everything else (6 algos).

Eval dir tags: `og50` = cells (1,1),(10,10),(25,25),(50,50),(100,100) + Random
c∈{10,25,50,100} (cube-double-play also (5,5)); `og50k5` = (5,5) + Random c=5;
`og50r1` = Random c=1 (only where needed); `og50fx` = all fixed bank policies on
the same episodes (oracle U). `report_og50.py --extra_tags` merges the first three;
`make_paper_assets.py --oracle_tag` reads the last.

---

## 3. Pipeline, stage by stage

Every stage is idempotent and writes to a distinct directory. Costs are observed
wall-clock (size `--time` at 2–4× the observed max, never a generic 12 h).

### 3.1 Policy bank (GPU, 3–12 h per row)
Rows live in `scripts/*_manifest.csv` (`idx,env_name,algo,seed,flags`). Flags are
the OGBench defaults per family (e.g. cube: gcivl α=10, gciql α=0.03, qrl α=0.03,
crl α=0.1, hiql 3/3/subgoal 10). Submit:
```bash
RUN_GROUP=mpilot SAVE_DIR=/scratch/jwquan/wmpp/policies/mpilot \
sbatch --array=91-108 --time=12:00:00 --exclude=fc10507,fc10512,fc11013 \
  scripts/slurm_train_array.sbatch scripts/puzzle_expand_manifest.csv
```
Outputs `params_1000000.pkl` + `eval.csv` (the last row = official 5×50 test
success at 1M steps = the baseline used in the paper). `scripts/p1_status.py`
tallies a manifest. For >12 h rows add `--save_interval=100000` to the flags.

### 3.2 World model + metric value head (GPU 2g/3g MIG, ~4–8 h)
```bash
sbatch -J wmpp-wm-<env> --time=8:00:00 --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 \
  scripts/slurm_train_wm.sbatch <env> <H> \
  --run_group=wm-value-lavl --save_dir=/scratch/jwquan/wmpp/wm --train_steps=1000000 \
  --wm.batch_size=1024 \
  --wm.lavl_discount=<g> --wm.lavl_expectile=<k> --wm.lavl_smoothness_weight=<sm>
```
Per-family settings used for every paper WM (all: E=3, MLP 512×3 + LayerNorm,
Adam 3e-4, batch 1024, 1M steps, latent 64, rep 10, τ=0.005, goals 0.2 cur /
0.5 traj-geometric / 0.3 random, horizon curriculum ramp 100k):

| family | H | γ | κ | smoothness |
|---|---|---|---|---|
| cube / puzzle | 5 | 0.99 | 0.7 | 0 |
| scene | 5 | 0.998 | 0.7 | 10 |
| antmaze-large | 10 | 0.999 | 0.9 | 10 |
| humanoidmaze-giant | 10 | 0.999 | 0.9 | 0 |

The run dir is `wm/wmogbench/wm-value-lavl/sd000_s_<jobid>.<timestamp>`; record
it in `manifests/wmpp_env_config.json` (`wm_dir`, `wm_epoch`; antmaze uses
900000 because the 1M checkpoint timed out).

### 3.3 Planner evaluation — the og50 protocol (CPU, 5 min–1 h per job)
One job per (env, bank seed). Template (copy from `launchers/launch_og50.sh`):
```bash
sbatch --account=def-gigor_cpu --cpus-per-task=8 --mem=24G --time=4:00:00 \
  --exclude=fc30554,fc30560,fc30572,fc30657 -J wmpp-og50-<env>-sd<s> \
  -o /scratch/jwquan/wmpp/logs/%x_%j.out --wrap="
set -euo pipefail
source /scratch/jwquan/wmpp/venv/bin/activate
export OGBENCH_IMPLS=/project/6067317/jwquan/ogbench/impls MUJOCO_GL=disable JAX_PLATFORMS=cpu
cd /project/6067317/jwquan/wm-policy-portfolio
python scripts/eval_planner.py --env_name=<env> --wm_dir=<wm_dir> --wm_epoch=1000000 \
  --policy_root=<root> --policy_epoch=1000000 --seeds=<s> --best_fixed=<best_fixed_per_seed[s]> \
  --horizon=100 --commit=100 --score_mode=value --score_agg=max \
  --kc_sweep=5,10,25,50 \
  --variants=score1_commit1,score5_commit5,score10_commit10,score25_commit25,score50_commit50,score100_commit100 \
  --random_commit=1,5,10,25,50,100 --random_seed=0 \
  --episodes_per_task=50 --skip_fixed --out_tag=og50"
```
Notes: `--best_fixed` only labels the summary (no fixed policies are run with
`--skip_fixed`); take it from `best_fixed_per_seed` in the env config.
`--variants=none` runs Random only. Humanoid needs `--time=8:00:00` (4000-step
episodes). Walltime anchors: cube/scene/puzzle-3x3 ≈ 5–15 min, puzzle-4x5/4x6 and
cube-triple ≈ 25–40 min, humanoid ≈ 50 min for a single cell; the full 6-cell +
6-Random job ≈ 3–6× that.

### 3.4 Report
```bash
python scripts/report_og50.py --envs=all --onestep_as_cell --extra_tags=og50r1,og50k5 \
  --fixed_tag=og50fx --out=/scratch/jwquan/wmpp/planner_eval/og50_final_sweep
```
→ `og50_final_sweep.md` (human tables: main, k=c sweep, Random-at-every-c,
switching/latency, absolute success incl. every policy) and `.json` (everything
the paper assets need). A new env with all cells in one `og50` dir needs no
`--extra_tags`; the flag is harmless if the extra dirs are absent.

### 3.5 Paper assets
```bash
python scripts/make_paper_assets.py      # writes WMPP_ICLR2027/{tables,figures}/
cd WMPP_ICLR2027 && latexmk -pdf main.tex
```
Tables/figures are `\input`/`\includegraphics` from `main.tex`; `tables/numbers.tex`
defines macros (`\MeanWMPP`, `\NumSigUp`, `\GainScenePlay`, …) used in the
running text, so **never hand-edit a number in main.tex** — regenerate.
To ship to Overleaf: `zip -r WMPP_ICLR2027_<date>.zip WMPP_ICLR2027 -x '*.aux' '*.log' '*.out' '*.bbl' '*.blg' '*.fls' '*.fdb_latexmk' '*.synctex.gz'`.

### 3.6 Tests
```bash
JAX_PLATFORMS=cpu MUJOCO_GL=disable OGBENCH_IMPLS=/project/6067317/jwquan/ogbench/impls \
  python -m pytest -q tests/test_planners.py tests/test_world_model.py
```
`tests/test_snapshot_determinism.py` and `tests/test_frozen_policy_restore.py`
need checkpoints on scratch and take minutes; run them after touching
`interfaces/` or `evaluation/`.

---

## 4. Job hygiene and coordination (mandatory)

1. **Before submitting anything**, check `squeue -u jwquan -o "%.10i %.40j %.8T %.10M %R"`
   *and* `sacct -X -u jwquan --starttime=<today-2d> --format=JobID,JobName%40,State`
   for a job with the same name — a parallel session once duplicated a trio of
   evals. Job names are the coordination key: `wmpp-<tag>-<env>-sd<s>`.
2. Append every submission (job ids, name, purpose) to
   `/scratch/jwquan/wmpp/launchers/SUBMISSIONS.log` (one line per job) and note
   it in the memory file if you are an agent with memory.
3. Walltime = 2–4× the observed elapsed of the closest comparable job
   (`sacct -X --format=JobID,JobName%40,Elapsed,State`). Pending TimeLimit can
   be shortened, never extended.
4. Bad nodes: CPU `fc30554,fc30560,fc30572,fc30657` (node-local /scratch mount
   faults → venv imports fail in seconds); GPU `fc10507` (prolog), `fc10512`
   (silent CPU fallback), `fc11013` (CUDA init). Always pass the exclude lists.
5. A job that dies within ~10 s with `ModuleNotFoundError`/`jaxlib` errors is a
   node fault: resubmit with the node excluded, do not debug the code.
6. `sacct` defaults to since-midnight — pass `--starttime`.
7. Never delete or rewrite anything under `planner_eval/*/bank_sd*_og50*`;
   add a new `--out_tag` instead.
8. Recover an exact past command with `sacct -j <id> -X --format=SubmitLine%800 -P`.
9. Do not redirect `TMPDIR` to /scratch in jobs (XLA compile temps hit quota).

---

## 5. State of every experiment (2026-08-25)

### 5.1 Complete (19 datasets, og50 protocol, 3 bank seeds)
antmaze-large-navigate, humanoidmaze-giant-navigate, cube-{single,double,triple,quadruple}-{play,noisy},
scene-{play,noisy}, puzzle-{3x3,4x4,4x5}-{play,noisy}, puzzle-4x6-play.
Numbers: `og50_final_sweep.md`. Main table with ±std over seeds:
`planner_eval/og50_main_table.tex` / `WMPP_ICLR2027/tables/main_results.tex`.

### 5.2 Pending: puzzle-4x6-noisy-v0
- Bank: array `54662964` rows 91–108 (`puzzle_expand_manifest.csv`, RUN_GROUP=mpilot):
  rows 91–97 COMPLETED, 98–108 PENDING (Priority). Nothing is wrong; it is fair-share.
- WM: DONE — `wm/wmogbench/wm-value-lavl/sd000_s_54662967.*` @ 1M (H=5, 0.99/0.7/sm0).
- When all 18 rows are COMPLETED:
  1. `python scripts/make_mpilot_manifest.py`-style bookkeeping is not needed; just
     compute per seed the best family from each run's `eval.csv` last row and add
     the env to `manifests/wmpp_env_config.json` (copy the puzzle-4x5-noisy entry
     as a template: `policy_root=/scratch/jwquan/wmpp/policies/mpilot/OGBench/mpilot`,
     `wm_dir`, `wm_epoch=1000000`, `best_fixed_per_seed`, `ogbench_test_success`
     for all 18 checkpoints, `ogbench_test_family_mean`, `best_policy_ogbench_test`, `bank_size`).
  2. Submit 3 og50 jobs (template §3.3, `--time=4:00:00`, all six cells + Random at
     all six c in one job, `--out_tag=og50`).
  3. `report_og50.py` (§3.4) → 20-env report; `make_paper_assets.py`; recompile;
     update the abstract/intro sentence that says "19 of 20".
  4. The static-oracle column (§5.4) for this env needs one extra job per seed
     that runs the fixed bank on the same og50 episodes: template §3.3 but
     `--variants=none --episodes_per_task=50 --out_tag=og50fx`, **without**
     `--skip_fixed` and without `--random_commit` (see `launchers/launch_og50fx.sh`).

### 5.2b New: pointmaze-medium-navigate-v0 (user request 2026-08-25, "more maze envs")
- Bank: array `56830805` (18 rows, `scripts/pointmaze_manifest.csv`, RUN_GROUP=mpilot,
  OGBench default hypers for pointmaze-medium; 6 h on 3g).
- WM: `56830806`, H=10, LAVL value-head hypers from `scripts/lavl_hyper.sh`
  (pointmaze-medium-navigate: γ=0.999, κ=0.9, smoothness 10), 1M steps, 3g 12 h.
- When both are COMPLETED: `python scripts/onboard_env.py --env_name=pointmaze-medium-navigate-v0
  --policy_root=/scratch/jwquan/wmpp/policies/mpilot/OGBench/mpilot --wm_dir=<run dir> --wm_epoch=1000000`
  writes the env-config entry and prints the og50 + og50fx sbatch lines; submit them,
  then §3.4–3.5. `make_paper_assets.py` already lists pointmaze under the Maze family.
- Candidates after this one (user intent): other pointmaze/antmaze sizes and stitch datasets.

### 5.3 Superseded / on disk only
- 20-episode protocol runs: tags `lavlp`, `lavlp_h1` (100,1), `abl`, `lavlv*`,
  `hsweep`, `2x2*`, `value*`, `vhsweep`; aggregates `aggregate_*.json`;
  `ablation_abl.{md,json}` (One-Step/Random ablation under the old protocol).
- WM horizon ablation (H=5/50/100 on puzzle-4x4-play/noisy, cube-triple-play):
  second-order, non-monotone; H=5 kept. Appendix material only if space.
- score_agg ablation (max/mean/last on cube-double-play): all CIs overlap; max kept.
- Diagnostics vs the dynamic oracle (`oracle/<env>/diagnostics_lavl.json`, 7 envs):
  ranking accuracy of the learned scorer vs realized outcomes; cube-double-play
  value ranking 0.62@H=5 → 0.84@H=100.
- Visual policy bank (`policies/visual`): out of scope.

### 5.4 Analyses derived from existing data (no new jobs)
`make_paper_assets.py` computes the **static per-episode oracle**
U = P(at least one fixed policy succeeds on the episode) from runs in which every
bank member was executed on identical episodes. Source of truth: tag **`og50fx`**
(fixed policies only, official 5×50 protocol, same episode seeds as og50; jobs
56816927–56816987 submitted 2026-08-25, launcher `launchers/launch_og50fx.sh`).
The same `og50fx` runs are also the paper's fixed-policy baselines (Table 1,
App. C bank table, `B_max`/`D_gap` in App. E) and the fixed-policy ms/step anchor
(App. F) — pass `--fixed_tag=og50fx` to `report_og50.py` (§3.4). The `abl`
fallback in `make_paper_assets.py` is unused now that all 57 jobs are complete
(`\NumOracleFallback` = 0); the macros `\OracleEpsPerGoal` / `\OracleEpisodes` follow.
Result (og50fx, all 57 jobs COMPLETED 2026-08-25): WMPP exceeds U by ≥1 point on
**6/19** datasets (cube-double-play 72.8 vs U 56.7; cube-double-noisy 60.4 vs 31.3;
scene-play 91.2 vs 59.1; scene-noisy 70.7 vs 51.7; cube-triple-noisy 18.8 vs 11.2;
puzzle-4x4-noisy 53.6 vs 51.6) — within-episode switching composes behaviour no
single policy exhibits; numerically above U on 8 (two are trivial floor/saturation
margins, hence the 1-point rule in the paper). `U − best` explains the null rows
(puzzle-4x5-noisy: 0.4 pt headroom). Table: `WMPP_ICLR2027/tables/bank_stats.tex`.

---

## 6. Known caveats to keep stating honestly

- `(k,c)` is selected per dataset by test success over the 3 seeds (there is no
  separate validation split under the official protocol). The full sweep is
  reported (appendix) so sensitivity is visible; say this in the paper.
- Random-Switch at its *best* c beats WMPP on 6/19 datasets when c is chosen
  post hoc (puzzle-3x3-play c=5: 43.5 vs 20.4; 3x3-noisy 97.7 vs 93.9;
  cube-triple-noisy 22.4 vs 18.8; cube-triple-play, cube-quadruple-noisy, 4x5-play
  by ≤1 pt). Fast random cycling is an anti-stuck/ensemble effect WMPP's long-k
  cells do not exploit. Report the Random-at-every-c table; do not hide it.
- cube-single-play is the one loss > 1 pt (−3.7, paired CI [−9.6, +2.3], n.s.): a single
  dominant policy (GCIVL 49.3, next best 12.1); any switch away costs.
- Floors (cube-quadruple-*, cube-triple-play, humanoid) and saturation
  (cube-single-noisy) are null by construction; keep them in the table.
- Latency: WMPP adds ≈ +2–4 ms per env step on CPU (fixed policy ≈ 0.2 ms) at
  P·E ≈ 15–18 model calls per step; every-step replanning at k=100 would be
  100× that, which is why only the diagonal is used.

---

## 7. Open TODOs (ordered)

1. **puzzle-4x6-noisy** → §5.2 (blocked on the bank array; check `sacct -j 54662964`).
2. Paper polish items that need new runs (not started; ask the user before launching):
   - bank-composition ablation (top-2 / top-3 / drop-best / same-algorithm multi-seed) on
     cube-double-play, scene-noisy, puzzle-4x4-noisy, cube-single-play + one floor env;
   - qualitative timelines (policy chain, score traces) for 3 wins + 3 failures —
     data already exists in `episodes.csv:policy_chain` and could be plotted without new runs.
3. Initial git commit + tag (`git add -A && git commit -m "WMPP: og50 protocol, 19-env results, ICLR draft"`)
   so agents can use worktrees and the paper can cite a hash.
4. Anonymous code release checklist (strip absolute paths from defaults in
   `scripts/*.py`, add `requirements.txt`).

---

## 8. Quick recipes

- Absolute success table for one env from raw runs:
  `python - <<EOF` … read `planner_eval/<env>/bank_sd{0,1,2}_<tag>/summary.json['success']`, average per method across seeds.
- Which cell/variant produced a number: `og50_final_sweep.json[i]['methods']['WMPP']['variant']`.
- Per-episode pairing key: `(task_id, episode_idx)`; `reset_seed` column is the env seed.
- Policy chain grammar in `episodes.csv`: `name:steps>name:steps>…`.
- Live progress: `tail -f /scratch/jwquan/wmpp/logs/<jobname>_<jobid>.out`.

## 9. Repo hygiene

No commit exists yet (`git status` shows everything untracked). Before agents
work in parallel: commit, then use `git worktree` or feature branches; keep
`WMPP_ICLR2027.zip` (the user's Overleaf export of 2026-08-25) untracked or
delete it once `WMPP_ICLR2027/` is authoritative. `.gitignore` already excludes
`wandb/`, `exp/`, `*.pkl`, `*.npz`, `logs/`.
