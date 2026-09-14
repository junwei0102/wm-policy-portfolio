# wm-policy-portfolio — agent instructions

Read `RUNBOOK.md` first; it is the source of truth for the protocol, paths,
locked decisions, job hygiene and the current state of every experiment.

Hard rules:
- Never modify `/project/6067317/jwquan/ogbench` (pinned OGBench v1.2.1).
- Never overwrite or delete `planner_eval/*/bank_sd*_og50*` runs; use a new `--out_tag`.
- Check `squeue`/`sacct` for a same-named job before submitting; log submissions
  in `/scratch/jwquan/wmpp/launchers/SUBMISSIONS.log`.
- CPU evals: `--account=def-gigor_cpu --exclude=fc30554,fc30560,fc30572,fc30657`;
  walltime = 2–4× observed elapsed of the closest prior job.
- Paper numbers are generated: run `scripts/make_paper_assets.py`, never hand-edit
  numbers in `WMPP_ICLR2027/main.tex`.
- Environment: `source /scratch/jwquan/wmpp/venv/bin/activate`;
  `export OGBENCH_IMPLS=/project/6067317/jwquan/ogbench/impls MUJOCO_GL=disable JAX_PLATFORMS=cpu`.

Useful tips:
- When explaining the results, code, decisions, plans or any other questions, use the most understandable and simple language. Reply in Simplified Chinese.