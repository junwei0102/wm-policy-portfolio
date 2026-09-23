"""Offline value-head criterion from CROSSING trajectory pairs in the 100M OGBench datasets.

The arbiter compares, from one state, the places different policies reach. The triplet criterion
(value_head_criterion.py) could not test that: its three states lie on one path. In the 100M play
datasets many trajectories pass through the same task-relevant state and then diverge, which gives
real branches from a shared start:

  head  : s_i (episode A) and s_j (episode B) have the same task configuration (the success
          predicate's discretisation) and an arm pose within --arm_tol normalised units;
  branch: the two episodes run k steps to s_a = A[t_i+k], s_b = B[t_j+k], with different task
          configurations (otherwise the branches differ only in the arm pose);
  tail  : both episodes later reach a common task configuration g; the goal observation is A's
          first arrival there, and the label says whose branch state is closer in time to it:
          r_a = arrival_A - (t_i+k), r_b = arrival_B - (t_j+k), label = 1[r_a < r_b] (ties 1/2).

score_h(pair) = 1[V_h(s_closer, g) > V_h(s_farther, g)]; concordance A_h = mean score, with a
cluster bootstrap over the head episode A. `imagined` replaces s_a, s_b by the world model's
k-step predictions from s_i, s_j under the logged actions (the arbiter's actual input).

  python scripts/crossing_criterion.py --env_name=scene-play-v0 --pairing=random --label=firstarrival \
      --chunk_start=50 --n_chunks=1 --min_margin=20 --n_queries=200000      # the paper's table
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from absl import app, flags  # noqa: E402

IMPLS = os.environ.get('OGBENCH_IMPLS', '/path/to/ogbench/impls')
if IMPLS not in sys.path:
    sys.path.insert(0, IMPLS)

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'scene-play-v0', 'OGBench dataset (its 100M chunks are read from --data_root).')
flags.DEFINE_string('data_root', '/path/to/ogbench_100m', 'Root holding <env>-100m-v0/<env>-NNN.npz chunks (the held-out sample: one 1M chunk of the OGBench 100M release).')
flags.DEFINE_integer('n_chunks', 10, 'Training chunks to load (1M transitions each).')
flags.DEFINE_integer('chunk_start', 0, 'First chunk index (e.g. 50 with --n_chunks=1: one held-out 1M sample, the size of the training set).')
flags.DEFINE_string('pairing', 'crossing', 'crossing: k-step branches from a shared start state; random: two states of different trajectories '
                    '(no shared start; pair difficulty is then controlled by the label margin, reported per band).')
flags.DEFINE_string('margin_bands', '', 'Comma list of margin band edges for the per-band report, e.g. "20,50,100" (label units).')
flags.DEFINE_string('config', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'manifests', 'wmpp_env_config.json'), 'Env registry.')
flags.DEFINE_string('wm_dir', None, 'World model (metric head); default: the registry entry.')
flags.DEFINE_string('policy_root', None, 'Bank root (direct head = its GCIQL member); default: the registry entry.')
flags.DEFINE_string('seeds', '0,1,2', 'Bank seeds: one direct head (GCIQL member) per seed.')
flags.DEFINE_integer('k', 10, 'Branch length in env steps (the family interval).')
flags.DEFINE_float('arm_tol', 1.0, 'Head tolerance on the proprioceptive dims, in per-dim std units.')
flags.DEFINE_integer('min_margin', 0, 'Keep pairs with |r_a - r_b| >= this many steps (0 = all; ties count 1/2).')
flags.DEFINE_integer('max_r', 400, 'Skip arrivals farther than this from the branch state (labels from random play get noisier with distance).')
flags.DEFINE_integer('n_queries', 20000, 'Head states sampled.')
flags.DEFINE_integer('goals_per_pair', 2, 'Common future configurations used per crossing pair.')
flags.DEFINE_string('variants', 'traj,imagined', 'Comma list.')
flags.DEFINE_string('goal_source', 'dataset', 'minpath only. dataset: any configuration reached from both branch configurations anywhere '
                    'in the data; episode: only configurations that BOTH crossing episodes reach later (tail crossing).')
flags.DEFINE_string('label', 'minpath', 'firstarrival: whose own episode reaches g sooner after the branch (noisy under random play); '
                    'minpath: the dataset-wide shortest observed time from each branch configuration to g (min over every '
                    'episode and occurrence), an empirical distance-to-goal between task configurations; '
                    'bfs (puzzle only): the exact press distance on the button graph, with goals drawn from anywhere in the dataset.')
flags.DEFINE_integer('n_boot', 2000, '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_string('out', '', 'JSON output path.')

PROPRIO = 19


def task_key(env, obs):
    """Task-relevant configuration id per state, discretised at the success tolerance (see world_model/success_predicates.py)."""
    if env.startswith('scene'):
        cube = np.round(obs[:, 19:22] / 0.8).astype(np.int64) + 10          # cells of 2x the 0.4 cube tolerance
        btn = obs[:, 28:30].argmax(1) * 2 + obs[:, 32:34].argmax(1)
        dr = np.round(obs[:, 36] / (0.04 * 18 * 2)).astype(np.int64) + 20
        wi = np.round(obs[:, 38] / (0.04 * 15 * 2)).astype(np.int64) + 20
        return ((btn * 64 + dr) * 64 + wi) * 10 ** 6 + (cube[:, 0] * 10000 + cube[:, 1] * 100 + cube[:, 2])
    if env.startswith('puzzle'):
        n = int(env.split('-')[1].split('x')[0]) * int(env.split('-')[1].split('x')[1])
        bits = obs[:, PROPRIO:PROPRIO + 4 * n].reshape(-1, n, 4)[:, :, :2].argmax(-1)
        return (bits * (1 << np.arange(n))).sum(1)
    if env.startswith('cube'):
        n = dict(single=1, double=2, triple=3, quadruple=4)[env.split('-')[1]]
        cells = np.stack([np.round(obs[:, 19 + 9 * i:22 + 9 * i] / 0.8).astype(np.int64) + 10 for i in range(n)], 1)  # 2x the 0.4 tolerance
        key = np.zeros(len(obs), np.int64)
        for i in range(n):
            key = key * 10 ** 6 + cells[:, i, 0] * 10000 + cells[:, i, 1] * 100 + cells[:, i, 2]
        return key
    raise NotImplementedError(env)


def lights_out_distance(env):
    """Exact press distance on the puzzle's button graph. A press toggles the button and its 4-neighbours
    (puzzle_env.py), i.e. XORs a fixed pattern, so d(s, g) = d(s ^ g, 0): one BFS from the all-off state."""
    rows, cols = (int(x) for x in env.split('-')[1].split('x'))
    n = rows * cols
    pats = []
    for i in range(n):
        x, y = i // cols, i % cols
        m = 0
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            if 0 <= x + dx < rows and 0 <= y + dy < cols:
                m |= 1 << ((x + dx) * cols + (y + dy))
        pats.append(m)
    dist = np.full(1 << n, -1, np.int32)
    dist[0] = 0
    frontier = [0]
    while frontier:
        nxt = []
        for c in frontier:
            for m in pats:
                d = c ^ m
                if dist[d] < 0:
                    dist[d] = dist[c] + 1
                    nxt.append(d)
        frontier = nxt
    return dist


def load_chunks(env, root, n, start=0):
    d = os.path.join(root, f'{env[:-3]}-100m-v0')
    O, A, EP, T = [], [], [], []
    nep = 0
    for c in range(start, start + n):
        z = np.load(os.path.join(d, f'{env[:-3]}-v0-{c:03d}.npz'))
        o, a, term = z['observations'], z['actions'], z['terminals']
        ends = np.flatnonzero(term) + 1
        starts = np.r_[0, ends[:-1]]
        for s, e in zip(starts, ends):
            O.append(o[s:e]); A.append(a[s:e]); EP.append(np.full(e - s, nep, np.int64)); T.append(np.arange(e - s)); nep += 1
    return np.concatenate(O), np.concatenate(A), np.concatenate(EP), np.concatenate(T), nep


def concordance(v_close, v_far):
    s = np.where(v_close > v_far, 1.0, np.where(v_close < v_far, 0.0, 0.5))
    return s, float(np.mean(v_close == v_far))


def cluster_boot(scores, cluster, n_boot, rng):
    uniq = np.unique(cluster)
    idx = {c: np.flatnonzero(cluster == c) for c in uniq}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, len(uniq), replace=True)
        boot[b] = np.concatenate([scores[idx[c]] for c in picked]).mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main(_):
    from interfaces.frozen_policy import FrozenPolicy  # noqa: E402
    from interfaces.policy_bank import find_run_dirs  # noqa: E402
    from world_model.model import EnsembleWorldModel  # noqa: E402

    env = FLAGS.env_name
    cfg = json.load(open(FLAGS.config))[env]
    rng = np.random.default_rng(FLAGS.seed)
    k = FLAGS.k
    obs, act, ep, t, nep = load_chunks(env, FLAGS.data_root, FLAGS.n_chunks, FLAGS.chunk_start)
    key = task_key(env, obs)
    ep_start = np.flatnonzero(np.r_[True, np.diff(ep) != 0])
    ep_end = np.r_[ep_start[1:], len(obs)]
    print(f'[cross] {env}: {nep} episodes, {len(obs)} steps, {len(np.unique(key))} task configurations, k={k}', flush=True)

    # ---- head partners: same task configuration, another episode, nearest arm pose
    z = obs[:, :PROPRIO] / (obs[:, :PROPRIO].std(0) + 1e-6)
    order = np.argsort(key, kind='stable')
    skey = key[order]
    bounds = np.searchsorted(skey, np.unique(skey))
    bounds = np.r_[bounds, len(skey)]
    lo_of = {skey[b]: (b, e) for b, e in zip(bounds[:-1], bounds[1:])}
    # ---- empirical configuration distance: shortest observed time from leaving p to first reaching q, over the dataset
    ukey, cid = np.unique(key, return_inverse=True)
    C = len(ukey)
    dkeys = dvals = None
    if FLAGS.label == 'minpath':
        # every (earlier segment p, later segment q) of an episode contributes start_q - (end_p - 1); the minimum over
        # all later occurrences of a configuration equals the time to its first arrival. Sparse: only observed pairs.
        buf_i, buf_d, parts = [], [], []

        def flush():
            if not buf_i:
                return
            kk, dd = np.concatenate(buf_i), np.concatenate(buf_d)
            o = np.argsort(kk, kind='stable'); kk, dd = kk[o], dd[o]
            u, first = np.unique(kk, return_index=True)
            parts.append((u, np.minimum.reduceat(dd, first)))
            buf_i.clear(); buf_d.clear()
        for n_done, (a, b) in enumerate(zip(ep_start, ep_end)):
            c = cid[a:b]
            seg = np.flatnonzero(np.r_[True, np.diff(c) != 0])
            seg_end = np.r_[seg[1:], b - a]
            pi, qi = np.triu_indices(len(seg), k=1)
            buf_i.append(c[seg[pi]].astype(np.int64) * C + c[seg[qi]])
            buf_d.append((seg[qi] - (seg_end[pi] - 1)).astype(np.float32))
            if len(buf_i) >= 5000:
                flush()
        flush()
        kk, dd = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        o = np.argsort(kk, kind='stable'); kk, dd = kk[o], dd[o]
        dkeys, first = np.unique(kk, return_index=True); dvals = np.minimum.reduceat(dd, first)
        del parts, kk, dd
        print(f'[cross] empirical distances: {len(dkeys)} configuration pairs observed among {C} configurations', flush=True)

    def dmin_lookup(p, q):
        i = np.searchsorted(dkeys, p * C + q)
        return float(dvals[i]) if i < len(dkeys) and dkeys[i] == p * C + q else np.inf
    bfs = lights_out_distance(env) if FLAGS.label == 'bfs' else None
    if bfs is not None:
        # presses only reach 2^rank of the configurations; the environment starts from random button states, so the
        # data spans several cosets of that subgroup. A goal is only meaningful inside the coset of the branch state.
        reach = np.flatnonzero(bfs >= 0)
        coset = np.full(len(bfs), -1, np.int32)
        n_coset = 0
        for c in range(len(bfs)):
            if coset[c] < 0:
                coset[c ^ reach] = n_coset
                n_coset += 1
        state_coset = coset[key]
        coset_states = {c: np.flatnonzero(state_coset == c) for c in range(n_coset)}
        print(f'[cross] lights-out BFS: {len(reach)} of {len(bfs)} configurations reachable per coset, {n_coset} cosets, '
              f'max distance {bfs.max()}', flush=True)
    queries = rng.choice(len(obs), FLAGS.n_queries, replace=False)
    pairs = []  # (i, j, g_idx_in_A, label, r_a, r_b)
    n_head = n_branch = 0
    partners = rng.integers(0, len(obs), len(queries))  # random pairing only
    if FLAGS.pairing == 'random' and FLAGS.label == 'firstarrival':
        # Simplest form: s_a and a later goal g on one trajectory; s_b on another trajectory that also reaches g
        # (the environment's success predicate); labels are each trajectory's own time to reach g. The cell index
        # only narrows the candidate states; reaching is decided by the exact predicate.
        from world_model.success_predicates import cube_success, puzzle_success, scene_success
        if env.startswith('scene'):
            succ = scene_success
        elif env.startswith('cube'):
            succ = lambda o, g: cube_success(dict(single=1, double=2, triple=3, quadruple=4)[env.split('-')[1]], o, g)
        else:
            succ = lambda o, g: puzzle_success(int(env.split('-')[1].split('x')[0]) * int(env.split('-')[1].split('x')[1]), o, g)
        for i in queries:
            hi_ = ep_end[ep[i]]
            if i + 2 >= hi_:
                continue
            gi = rng.integers(i + 1, min(hi_, i + 1 + FLAGS.max_r))
            r_a = gi - i
            b, e = lo_of[key[gi]]
            m = order[b:e]
            m = m[ep[m] != ep[i]]
            if len(m) == 0:
                continue
            m = m[succ(obs[m], np.broadcast_to(obs[gi], (len(m), obs.shape[1]))) > 0]
            if len(m) == 0:
                continue
            eps_b = np.unique(ep[m])
            B = eps_b[rng.integers(0, len(eps_b))]
            t_reach = m[ep[m] == B].min()                                  # first state of B inside the goal set
            if t_reach - ep_start[B] < 2:
                continue
            ib = rng.integers(ep_start[B], min(t_reach, ep_start[B] + max(t_reach - ep_start[B], 1)))
            ib = rng.integers(max(ep_start[B], t_reach - FLAGS.max_r), t_reach)
            r_b = t_reach - ib
            if succ(obs[ib:ib + 1], obs[gi:gi + 1])[0] > 0 or abs(r_a - r_b) < FLAGS.min_margin:
                continue
            n_head += 1; n_branch += 1
            label = 1.0 if r_a < r_b else (0.0 if r_b < r_a else 0.5)
            pairs.append((i, ib, gi, label, r_a, r_b))
        queries = np.array([], dtype=np.int64)
    for qn, i in enumerate(queries):
        if FLAGS.pairing == 'random':
            ia, ib = i, partners[qn]
            if ep[ia] == ep[ib] or key[ia] == key[ib]:
                continue
            n_head += 1; n_branch += 1
        else:
            if t[i] + k >= ep_end[ep[i]] - ep_start[ep[i]]:
                continue
            b, e = lo_of[key[i]]
            m = order[b:e]
            m = m[ep[m] != ep[i]]
            if len(m) == 0:
                continue
            dd = np.linalg.norm(z[m] - z[i], axis=1)
            j = m[dd.argmin()]
            if dd.min() > FLAGS.arm_tol:
                continue
            n_head += 1
            ia, ib = i + k, j + k
            if ib >= ep_end[ep[j]] or key[ia] == key[ib]:
                continue  # branch not distinct in the task-relevant dims
            n_branch += 1
        if bfs is not None:
            # exact distances need no tail crossing: any dataset state serves as the goal observation
            pool = coset_states[state_coset[ia]]
            for gi in pool[rng.integers(0, len(pool), FLAGS.goals_per_pair)]:
                r_a, r_b = int(bfs[key[ia] ^ key[gi]]), int(bfs[key[ib] ^ key[gi]])
                if r_a < 0 or r_b < 0 or abs(r_a - r_b) < FLAGS.min_margin:
                    continue
                label = 1.0 if r_a < r_b else (0.0 if r_b < r_a else 0.5)
                pairs.append((ia, ib, gi, label, r_a, r_b))
            continue
        if FLAGS.label == 'minpath' and FLAGS.goal_source == 'dataset':
            # dataset-wide distances need no tail crossing of these two episodes: any configuration reached from BOTH
            # branch configurations somewhere in the data is a goal; its observation is a random state of that configuration
            pa, pb = cid[ia], cid[ib]
            la, ha = np.searchsorted(dkeys, [pa * C, (pa + 1) * C]); lb, hb = np.searchsorted(dkeys, [pb * C, (pb + 1) * C])
            qa, qb = dkeys[la:ha] - pa * C, dkeys[lb:hb] - pb * C
            common_q, ia_, ib_ = np.intersect1d(qa, qb, return_indices=True)
            keepq = (common_q != pa) & (common_q != pb)
            common_q, ra_all, rb_all = common_q[keepq], dvals[la:ha][ia_][keepq], dvals[lb:hb][ib_][keepq]
            if len(common_q) == 0:
                continue
            for qi in rng.choice(len(common_q), min(FLAGS.goals_per_pair, len(common_q)), replace=False):
                r_a, r_b = int(ra_all[qi]), int(rb_all[qi])
                if max(r_a, r_b) > FLAGS.max_r or abs(r_a - r_b) < FLAGS.min_margin:
                    continue
                gb, ge = lo_of[ukey[common_q[qi]]]
                gi = order[gb + rng.integers(0, ge - gb)]
                label = 1.0 if r_a < r_b else (0.0 if r_b < r_a else 0.5)
                pairs.append((ia, ib, gi, label, r_a, r_b))
            continue
        # first arrival of each episode at every configuration after its branch state (episode goals only)
        first_a, first_b = {}, {}
        for idx in range(ia + 1, ep_end[ep[ia]]):
            first_a.setdefault(key[idx], idx)
        for idx in range(ib + 1, ep_end[ep[ib]]):
            first_b.setdefault(key[idx], idx)
        common = [g for g in first_a if g in first_b and g != key[ia] and g != key[ib]]
        rng.shuffle(common)
        for g in common[:FLAGS.goals_per_pair]:
            if FLAGS.label == 'minpath':
                r_a, r_b = dmin_lookup(cid[ia], cid[first_a[g]]), dmin_lookup(cid[ib], cid[first_b[g]])
                if not (np.isfinite(r_a) and np.isfinite(r_b)):
                    continue
                r_a, r_b = int(r_a), int(r_b)
            else:
                r_a, r_b = first_a[g] - ia, first_b[g] - ib
            if max(r_a, r_b) > FLAGS.max_r or abs(r_a - r_b) < FLAGS.min_margin:
                continue
            label = 1.0 if r_a < r_b else (0.0 if r_b < r_a else 0.5)
            pairs.append((ia, ib, first_a[g], label, r_a, r_b))
    P = np.array(pairs, dtype=np.int64)
    print(f'[cross] queries {len(queries)}: head crossings {n_head}, distinct branches {n_branch}, labelled pairs {len(P)} '
          f'(ties {100 * np.mean(P[:, 3] == 0.5) if len(P) else 0:.1f}%, median |r_a - r_b| {np.median(np.abs(P[:, 4] - P[:, 5])) if len(P) else 0:.0f} steps)')
    if len(P) == 0:
        return
    labels = P[:, 3].astype(np.float64)
    keep = labels != 0.5  # exact ties in the label carry no information; report their share above
    print(f'[cross] label ties dropped: {int((~keep).sum())}', flush=True)
    P, labels = P[keep], labels[keep]
    close = np.where(labels == 1.0, P[:, 0], P[:, 1])
    far = np.where(labels == 1.0, P[:, 1], P[:, 0])
    goal = obs[P[:, 2]]
    cluster = ep[P[:, 0]]

    # ---- value heads
    wm_dir, policy_root = FLAGS.wm_dir or cfg['wm_dir'], FLAGS.policy_root or cfg['policy_root']
    wm = EnsembleWorldModel.load(wm_dir, cfg['wm_epoch'])
    E = int(wm.config['num_members'])
    heads = {'metric': lambda s, g: np.asarray(wm.value_score(np.broadcast_to(s, (E, *s.shape)), np.broadcast_to(g, (E, *g.shape)))).mean(0)}
    # reference scorers that know the task dims: how much of the label is recoverable at all
    if env.startswith('scene'):
        task_dims = [19, 20, 21, 28, 29, 32, 33, 36, 38]
    elif env.startswith('cube'):
        task_dims = [d for i in range(dict(single=1, double=2, triple=3, quadruple=4)[env.split('-')[1]]) for d in range(19 + 9 * i, 22 + 9 * i)]
    else:
        nb = int(env.split('-')[1].split('x')[0]) * int(env.split('-')[1].split('x')[1])
        task_dims = [PROPRIO + 4 * i + j for i in range(nb) for j in (0, 1)]  # button one-hots: euclid = sqrt(2 * hamming)
    sd = obs[:, task_dims].std(0) + 1e-6
    heads['ref/task_euclid'] = lambda s, g: -np.linalg.norm((s[:, task_dims] - g[:, task_dims]) / sd, axis=1)
    heads['ref/full_euclid'] = lambda s, g: -np.linalg.norm((s - g) / (obs.std(0) + 1e-6), axis=1)
    seeds = [int(x) for x in FLAGS.seeds.split(',')]
    runs = find_run_dirs(env, policy_root, cfg['policy_epoch'], algos=('gciql',), seeds=seeds)
    for name, run in sorted(runs.items()):
        pol = FrozenPolicy.load(run, cfg['policy_epoch'])
        heads[f'direct/{name}'] = (lambda p: (lambda s, g: np.asarray(p.value(s, g))))(pol)

    def imagine(start, steps):
        """WM prediction of s_{start+steps} replaying the logged actions from the real s_start (ensemble mean)."""
        x = np.broadcast_to(obs[start], (E, len(start), obs.shape[1]))
        for s in range(steps):
            x = np.asarray(wm.predict_next(x, np.broadcast_to(act[start + s], (E, len(start), act.shape[1]))))
        return x.mean(0)

    results = {}
    for variant in FLAGS.variants.split(','):
        if variant == 'traj':
            sc, sf = obs[close], obs[far]
        elif variant == 'imagined':
            if FLAGS.pairing == 'random':
                continue  # no branch to imagine
            sc, sf = imagine(close - k, k), imagine(far - k, k)
        else:
            raise ValueError(variant)
        entry = {}
        for name, fn in heads.items():
            vc, vf = fn(sc, goal), fn(sf, goal)
            s, tie = concordance(vc, vf)
            lo, hi = cluster_boot(s, cluster, FLAGS.n_boot, rng)
            entry[name] = dict(score=float(s.mean()), lo=lo, hi=hi, tie_rate=tie, scores=s)
        dnames = [n for n in entry if n.startswith('direct/')]
        dmean = np.mean([entry[n]['scores'] for n in dnames], axis=0)
        diff = dmean - entry['metric']['scores']
        lo, hi = cluster_boot(diff, cluster, FLAGS.n_boot, rng)
        results[variant] = {n: {kk: vv for kk, vv in e.items() if kk != 'scores'} for n, e in entry.items()}
        results[variant]['direct_minus_metric'] = dict(delta=float(diff.mean()), lo=lo, hi=hi)
        print(f'[cross] variant {variant}: {len(P)} pairs from {len(np.unique(cluster))} head episodes')
        for name, e in entry.items():
            print(f"   {name:16s} A = {e['score']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]   ties {100 * e['tie_rate']:.1f}%")
        print(f"   direct - metric  {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        if FLAGS.margin_bands:
            edges = [int(x) for x in FLAGS.margin_bands.split(',')] + [10 ** 9]
            marg = np.abs(P[:, 4] - P[:, 5])
            results[variant]['bands'] = {}
            for a_, b_ in zip(edges[:-1], edges[1:]):
                sel = (marg >= a_) & (marg < b_)
                if sel.sum() < 30:
                    continue
                dsel = diff[sel]; lo_, hi_ = cluster_boot(dsel, cluster[sel], FLAGS.n_boot // 2, rng)
                results[variant]['bands'][f'{a_}-{b_}'] = dict(n=int(sel.sum()), metric=float(entry['metric']['scores'][sel].mean()),
                                                            direct=float(dmean[sel].mean()), ref=float(entry['ref/task_euclid']['scores'][sel].mean()),
                                                            delta=float(dsel.mean()), lo=lo_, hi=hi_)
                print(f"      margin [{a_},{b_}): n={sel.sum():6d}  metric {entry['metric']['scores'][sel].mean():.3f}  direct {dmean[sel].mean():.3f}  "
                      f"ref {entry['ref/task_euclid']['scores'][sel].mean():.3f}  direct-metric {dsel.mean():+.3f} [{lo_:+.3f},{hi_:+.3f}]")
    meta = dict(env=env, label=FLAGS.label, goal_source=FLAGS.goal_source, pairing=FLAGS.pairing, chunk_start=FLAGS.chunk_start, n_chunks=FLAGS.n_chunks, k=k, arm_tol=FLAGS.arm_tol, min_margin=FLAGS.min_margin, max_r=FLAGS.max_r,
                n_queries=int(len(queries)), n_head=n_head, n_branch=n_branch, n_pairs=int(len(P)), n_head_episodes=int(len(np.unique(cluster))),
                median_margin=float(np.median(np.abs(P[:, 4] - P[:, 5]))))
    if FLAGS.out:
        json.dump(dict(meta=meta, results=results), open(FLAGS.out, 'w'), indent=1)
        print('wrote', FLAGS.out)


if __name__ == '__main__':
    app.run(main)
