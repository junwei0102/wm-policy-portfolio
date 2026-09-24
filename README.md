# WMPA: World-Model Policy Arbiter

Test-time arbitration over a bank of frozen goal-conditioned policies on
[OGBench](https://github.com/seohongpark/ogbench).

Several frozen goal-conditioned policies, trained by different offline GCRL
algorithms (GCBC, GCIVL, GCIQL, QRL, CRL, HIQL; one checkpoint per algorithm,
so a bank has six members; three bank seeds) on the *same* dataset, form a bank. Every `c` environment steps WMPA rolls each policy forward
`k` steps in a learned state-space world model, scores the imagined futures with
one shared goal-conditioned value function, and executes the highest-scoring
policy until the next arbitration. No policy is retrained. On 18 state-based
OGBench datasets (maze, cube, scene, puzzle) the macro-average success rate
rises from 44% (best policy in the bank) to 58%, with significant gains on 12
datasets; random switching at the same interval reaches 39%.

## Method

- **World model** (`world_model/model.py`): an ensemble of `E=3` one-step MLP
  delta-dynamics models in observation space, one model per dataset. Each policy
  acts closed-loop on its own imagined states.
- **Value**: for maze, cube, and scene the metric value
  `V(s,g) = -||φ_S(s) − φ_G(ψ(g))||` (LAVL head trained jointly with the
  dynamics); for puzzle the state value `V(s,g)` of the bank's GCIQL member.
- **Score and interval**: max over the horizon of the ensemble-mean value; the
  single interval `k = c` is set per task family (maze 1, cube 5, scene and
  puzzle 10) on held-out episodes. The budget is `M·E = 18` dynamics and value
  calls per environment step regardless of `k`.
- **Controls** (`planners/portfolio.py`, `planners/sim_rollout.py`): random
  switching at the same interval, Q-select (the GCIQL twin-Q critic, no model),
  action-level MPC on a single policy, stall-restart switching, own-value
  scoring, and the same arbitration with the real simulator in place of the model.

## Layout

```
interfaces/    FrozenPolicy (restores OGBench checkpoints), PolicyBank, simulator snapshot/restore
world_model/   ensemble dynamics + metric value head, sequence dataset, imagined rollouts, success predicates
planners/      WMPA RolloutRanker and the control arbiters
evaluation/    paired-episode harness, real-simulator oracle branches, ranking diagnostics
manifests/     wmpp_env_config.json: per-dataset registry (bank root, world-model checkpoint, per-seed priors)
scripts/       training, evaluation, and reporting entry points
```

## Setup

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/seohongpark/ogbench && (cd ogbench && git checkout 1d41409 && pip install -e ".[train]")
export OGBENCH_IMPLS=/path/to/ogbench/impls MUJOCO_GL=disable JAX_PLATFORMS=cpu   # GPU: drop JAX_PLATFORMS
```

Policies are trained with the unmodified OGBench reference implementations; this
repository only restores their checkpoints. Checkpoints, evaluation outputs, and
datasets are not included; every script takes its paths as flags
(`--policy_root`, `--wm_dir`, `--out_dir`, `--eval_root`).

## Pipeline

1. **Policy bank** (GPU): `scripts/train_policy.py` wraps `impls/main.py`, e.g.
   `--agent=agents/gciql.py --env_name=scene-play-v0 --seed=0`; one run per
   algorithm × seed × dataset with OGBench defaults.
2. **World model + metric value** (GPU): `scripts/train_world_model.py
   --env_name=... --train_steps=1000000` with the per-family config flags
   `--wm.horizon`, `--wm.discount`, `--wm.lavl_expectile`,
   `--wm.lavl_smoothness_weight` (defaults in `world_model/model.py:get_config`).
3. **Evaluation** (CPU, official protocol: 5 tasks × 50 episodes per bank seed,
   identical reset seeds for every method): `scripts/eval_planner.py
   --env_name=... --wm_dir=... --policy_root=... --seeds=0 --best_fixed=gciql-sd0
   --score_mode=value --score_agg=max --horizon=10 --commit=10
   --episodes_per_task=50 --out_tag=og50`. `--critic_scorer=gciql --critic_kc=10`
   scores with the direct value, `--kxc=10:5` / `--critic_kxc=10:5` any other
   $(k,c)$ cell; `--random_commit` and `--critic_select_commit`
   run the random and Q-select controls, `--bank_algos=gciql,gcivl,hiql
   --native_kc=<k> --native_select_commit=1,<k>` the own-value scoring and
   selection on the three-member bank, `--mpc_n=6 --mpc_sigma=0.2 --mpc_k=<k>
   --mpc_commit=1` the every-step action-level MPC; `--episode_range=50:100`
   evaluates the held-out episodes used for interval selection.
4. **Report**: `scripts/derive_family_cells.py` picks the per-family interval on
   the held-out episodes and prints the matching flags; `scripts/report_og50.py
   --fixed_tag=og50fx --select_rule=family
   --family_k=maze:1,cube:5,scene:10,puzzle:10
   --family_c=maze:1,cube:5,scene:10,puzzle:10 --family_scorer=puzzle:critic`
   aggregates per-dataset success and paired contrasts.
   `scripts/collect_oracle.py` and `scripts/run_wm_diagnostics.py` produce the
   real-simulator branch diagnostics.
5. **Value-head selection** (offline, from held-out trajectories):
   `scripts/crossing_criterion.py --env_name=... --pairing=random
   --label=firstarrival --chunk_start=50 --n_chunks=1 --min_margin=20` scores
   each head by whether it ranks higher the state whose trajectory reaches a
   shared goal sooner, on one held-out 1M chunk of the OGBench 100M release
   (`--data_root`).

## Statistics

Every contrast is paired per episode. Intervals are a hierarchical bootstrap
(resample bank seeds, then episodes within seed; 10⁴ resamples; 95% percentile
CI); a difference is significant when its interval excludes zero.
