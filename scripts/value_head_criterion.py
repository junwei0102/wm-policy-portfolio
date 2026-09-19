"""Offline criterion for choosing the value head, with no online rollouts.

Both heads are compared only through the ORDER they impose, so their different scales
(the metric head is a negative latent distance, the direct head an IQL value) do not matter.
From the benchmark's VALIDATION split we sample ordered triplets i < j < k inside one
trajectory, set the goal g = s_k, and score a head by how often it ranks the later state
as the closer one:

    score_h(i,j,k) = 1[V_h(s_j,g) > V_h(s_i,g)]  (exact ties count 1/2),
    A_h            = mean over N triplets,        chance level 1/2.

Three variants of what is scored (--variants):
  traj      real states, gap j-i in [1, k_family]: the arbitration scale. Minimal check.
  imagined  s_j is replaced by the world model's prediction of it, obtained by replaying the
            logged actions a_i..a_{j-1} from the real s_i, as at arbitration time. Isolates
            the effect of scoring model output instead of real states.
  neighbor  DIAGNOSTIC ONLY, weak label: the negative is the delta-step successor of the
            nearest neighbour of s_i taken from a different trajectory, assumed not to be
            closer to g. Closest to a real decision, but the label can be wrong.

Goals are future states of the same trajectory drawn geometrically, matching the value-goal
relabeling both heads were trained with; random goals from other trajectories are excluded
because the ordering label is undefined for them.

Uncertainty is a cluster bootstrap over trajectories (triplets from one trajectory move
together). The metric and direct heads see the SAME triplets, so their difference is paired.

  python scripts/value_head_criterion.py --env_name=cube-triple-play-v0
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from absl import app, flags  # noqa: E402

IMPLS = os.environ.get('OGBENCH_IMPLS', '/project/6067317/jwquan/ogbench/impls')
if IMPLS not in sys.path:
    sys.path.insert(0, IMPLS)

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'cube-triple-play-v0', 'OGBench dataset.')
flags.DEFINE_string('config', '/project/6067317/jwquan/wm-policy-portfolio/manifests/wmpp_env_config.json', 'Env registry.')
flags.DEFINE_string('wm_dir', None, 'World model (metric head); default: the registry entry.')
flags.DEFINE_string('policy_root', None, 'Bank root (direct head = its GCIQL member); default: the registry entry.')
flags.DEFINE_string('seeds', '0,1,2', 'Bank seeds: one direct head per seed.')
flags.DEFINE_integer('k_family', None, 'Arbitration interval of this family; default: the reported one (maze 1, cube 5, scene 10, puzzle 10).')
flags.DEFINE_integer('n_triplets', 10000, 'Triplets per variant.')
flags.DEFINE_string('variants', 'traj,imagined,neighbor', 'Comma list: traj, imagined, neighbor.')
flags.DEFINE_integer('n_boot', 2000, 'Cluster-bootstrap resamples.')
flags.DEFINE_integer('seed', 0, 'Sampling seed.')
flags.DEFINE_string('out', '', 'Optional JSON output path.')


def episodes(dataset):
    """List of (start, stop) index pairs, one per trajectory."""
    term = np.nonzero(np.asarray(dataset['terminals']) > 0)[0]
    out, prev = [], 0
    for t in term:
        if t + 1 - prev >= 4:
            out.append((prev, int(t) + 1))
        prev = int(t) + 1
    return out


def sample_triplets(eps, lengths, k_fam, discount, n, rng):
    """(traj, i, j, k) with i < j < k in one trajectory: gap j-i uniform in [1, k_fam],
    goal offset k-j geometric with the relabeling discount, truncated to the episode end."""
    out = []
    probs = np.asarray(lengths, dtype=np.float64)
    probs /= probs.sum()
    while len(out) < n:
        e = rng.choice(len(eps), p=probs)
        s0, s1 = eps[e]
        L = s1 - s0
        if L < k_fam + 3:
            continue
        i = s0 + rng.integers(0, L - k_fam - 2)
        j = i + 1 + rng.integers(0, k_fam)
        offs = min(int(rng.geometric(1 - discount)), s1 - 1 - j)
        if offs < 1:
            continue
        out.append((e, i, j, j + offs))
    return np.asarray(out, dtype=np.int64)


def concordance(v_i, v_j):
    """Per-triplet score with exact ties at 1/2, and the tie rate."""
    tie = v_j == v_i
    return np.where(tie, 0.5, (v_j > v_i).astype(np.float64)), tie.mean()


def cluster_boot(scores, traj, n_boot, rng):
    """95% interval of the mean, resampling trajectories (not triplets)."""
    uniq = np.unique(traj)
    idx = {t: np.nonzero(traj == t)[0] for t in uniq}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, len(uniq), replace=True)
        boot[b] = np.concatenate([scores[idx[t]] for t in picked]).mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main(_):
    from interfaces.frozen_policy import FrozenPolicy  # noqa: E402
    from interfaces.policy_bank import find_run_dirs  # noqa: E402
    from world_model.model import EnsembleWorldModel  # noqa: E402
    import ogbench  # noqa: E402

    cfg = json.load(open(FLAGS.config))[FLAGS.env_name]
    wm_dir = FLAGS.wm_dir or cfg['wm_dir']
    policy_root = FLAGS.policy_root or cfg['policy_root']
    FAMILY_K = dict(pointmaze=1, antmaze=1, cube=5, scene=10, puzzle=10)  # the intervals reported in the paper
    k_fam = FLAGS.k_family or FAMILY_K[FLAGS.env_name.split('-')[0]]
    rng = np.random.default_rng(FLAGS.seed)

    wm = EnsembleWorldModel.load(wm_dir, cfg['wm_epoch'])
    E = int(wm.config['num_members'])
    discount = float(wm.config['discount'])
    _, _, val = ogbench.make_env_and_datasets(FLAGS.env_name)
    obs = np.asarray(val['observations'], dtype=np.float32)
    act = np.asarray(val['actions'], dtype=np.float32)
    eps = episodes(val)
    lengths = [b - a for a, b in eps]
    print(f'[crit] {FLAGS.env_name}: {len(eps)} validation trajectories, '
          f'{np.mean(lengths):.0f} steps on average, k_family={k_fam}, geometric goals with discount {discount}')

    heads = {}
    heads['metric'] = lambda s, g: np.asarray(wm.value_score(
        np.broadcast_to(s, (E, *s.shape)), np.broadcast_to(g, (E, *g.shape)))).mean(axis=0)
    seeds = [int(x) for x in FLAGS.seeds.split(',')]
    runs = find_run_dirs(FLAGS.env_name, policy_root, cfg['policy_epoch'], algos=('gciql',), seeds=seeds)
    assert len(runs) == len(seeds), (sorted(runs), seeds)
    for name, run in sorted(runs.items()):
        pol = FrozenPolicy.load(run, cfg['policy_epoch'])
        heads[f'direct/{name}'] = (lambda p: (lambda s, g: np.asarray(p.value(s, g))))(pol)

    def imagine(start_idx, gaps):
        """World-model prediction of s_{i+gap} from the real s_i, replaying the logged actions."""
        x = np.broadcast_to(obs[start_idx], (E, len(start_idx), obs.shape[-1]))
        out = np.empty((len(start_idx), obs.shape[-1]), dtype=np.float32)
        done = np.zeros(len(start_idx), dtype=bool)
        for step in range(int(gaps.max())):
            a = np.broadcast_to(act[start_idx + step], (E, len(start_idx), act.shape[-1]))
            x = np.asarray(wm.predict_next(x, a))
            reached = (gaps == step + 1) & ~done
            if reached.any():
                out[reached] = x.mean(axis=0)[reached]
                done |= reached
        return out

    results = {}
    for variant in FLAGS.variants.split(','):
        tri = sample_triplets(eps, lengths, k_fam, discount, FLAGS.n_triplets, rng)
        traj, i, j, k = tri[:, 0], tri[:, 1], tri[:, 2], tri[:, 3]
        g = obs[k]
        if variant == 'traj':
            s_i, s_j = obs[i], obs[j]
        elif variant == 'imagined':
            s_i, s_j = obs[i], imagine(i, j - i)  # the later state is the model's prediction
        elif variant == 'neighbor':
            # weak label: the negative is the same-gap successor of a nearest neighbour of s_i
            # taken from another trajectory, assumed not to be closer to g.
            z = (obs - wm.normalizer['obs_mean']) / np.asarray(wm.normalizer['obs_std'])
            traj_id = np.full(len(obs), -1, dtype=np.int64)
            for e, (a0, b0) in enumerate(eps):
                traj_id[a0:b0] = e
            cand = rng.choice(len(obs) - k_fam - 1, size=(len(i), 64))
            same = traj_id[cand] == traj[:, None]  # a neighbour must come from another trajectory
            d = np.linalg.norm(z[cand] - z[i][:, None], axis=-1) + np.where(same, 1e6, 0.0)
            nn = cand[np.arange(len(i)), d.argmin(axis=1)]
            s_i, s_j = obs[nn + (j - i)], obs[j]
        else:
            raise ValueError(variant)
        entry = {}
        for name, fn in heads.items():
            sc, tie = concordance(np.asarray(fn(s_i, g), dtype=np.float64), np.asarray(fn(s_j, g), dtype=np.float64))
            lo, hi = cluster_boot(sc, traj, FLAGS.n_boot, np.random.default_rng(FLAGS.seed + 1))
            entry[name] = dict(score=float(sc.mean()), lo=lo, hi=hi, tie_rate=float(tie), scores=sc)
        dnames = [n for n in entry if n.startswith('direct/')]
        dmean = np.mean([entry[n]['scores'] for n in dnames], axis=0)
        diff = dmean - entry['metric']['scores']
        dlo, dhi = cluster_boot(diff, traj, FLAGS.n_boot, np.random.default_rng(FLAGS.seed + 2))
        results[variant] = {n: {kk: vv for kk, vv in e.items() if kk != 'scores'} for n, e in entry.items()}
        results[variant]['direct_minus_metric'] = dict(delta=float(diff.mean()), lo=dlo, hi=dhi)
        print(f'\n[{variant}] {len(tri)} triplets from {len(np.unique(traj))} trajectories')
        for name in sorted(entry):
            e = entry[name]
            print(f"   {name:16s} A = {e['score']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]   ties {100*e['tie_rate']:.1f}%")
        d = results[variant]['direct_minus_metric']
        print(f"   direct - metric  {d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
              f"  -> picks {'DIRECT' if d['delta'] > 0 else 'METRIC'}")
    if FLAGS.out:
        json.dump(results, open(FLAGS.out, 'w'), indent=1, default=float)
        print('\nwrote', FLAGS.out)


if __name__ == '__main__':
    app.run(main)
