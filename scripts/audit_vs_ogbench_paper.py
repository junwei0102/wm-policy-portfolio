"""Audit our policy bank against the published OGBench results table.

Extracts the per-dataset, per-algorithm success rates (mean +- std over the
paper's 8 seeds) straight from the OGBench paper PDF and compares them with the
`ogbench_test_family_mean` recorded in manifests/wmpp_env_config.json (each
checkpoint's own eval.csv at the loaded epoch, averaged over our 3 bank seeds).

A member is flagged when it sits further than max(3 sd, 8 points) from the
published value -- the band that caught the wrong-alpha banks in Aug 2026.
Reference numbers must come from the paper, never from memory.

Usage:
  python scripts/audit_vs_ogbench_paper.py [--pdf ogbench.pdf] [--json out.json]
Requires `pdftotext` (poppler) on PATH; stdlib only otherwise, so it runs on the
login node.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Column order of the paper's main results table (verify against its header row).
ALGOS = ['gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql']
FAMILIES = [
    ('Maze', ['pointmaze-medium-navigate-v0', 'antmaze-large-navigate-v0']),
    ('Cube', ['cube-single-play-v0', 'cube-single-noisy-v0', 'cube-double-play-v0', 'cube-double-noisy-v0',
              'cube-triple-play-v0', 'cube-triple-noisy-v0', 'cube-quadruple-play-v0', 'cube-quadruple-noisy-v0']),
    ('Scene', ['scene-play-v0', 'scene-noisy-v0']),
    ('Puzzle', ['puzzle-3x3-play-v0', 'puzzle-3x3-noisy-v0', 'puzzle-4x4-play-v0', 'puzzle-4x4-noisy-v0',
                'puzzle-4x5-play-v0', 'puzzle-4x5-noisy-v0', 'puzzle-4x6-play-v0', 'puzzle-4x6-noisy-v0']),
]
ENVS = [e for _, es in FAMILIES for e in es]

ROW = re.compile(r'\b([a-z0-9-]+-v0)\s+((?:\d+\s*±\s*\d+\s+){5}\d+\s*±\s*\d+)\s*$')
HEADER = re.compile(r'Dataset\s+GCBC\s+GCIVL\s+GCIQL\s+QRL\s+CRL\s+HIQL')


def paper_table(pdf_path):
    """{dataset: {algo: (mean, sd)}} from the paper's main results table."""
    with tempfile.TemporaryDirectory() as td:
        txt = os.path.join(td, 'paper.txt')
        subprocess.run(['pdftotext', '-layout', pdf_path, txt], check=True)
        lines = open(txt, errors='replace').read().splitlines()
    assert any(HEADER.search(l) for l in lines), (
        'column header "Dataset GCBC GCIVL GCIQL QRL CRL HIQL" not found; '
        'the table layout changed -- re-check ALGOS before trusting this audit')
    ref = {}
    for line in lines:
        m = ROW.search(line.rstrip())
        if not m:
            continue
        vals = [(int(a), int(b)) for a, b in re.findall(r'(\d+)\s*±\s*(\d+)', m.group(2))]
        if len(vals) == 6:
            ref.setdefault(m.group(1), dict(zip(ALGOS, vals)))  # first (main table) wins
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', default=os.path.join(ROOT, 'ogbench.pdf'))
    ap.add_argument('--config', default=os.path.join(ROOT, 'manifests', 'wmpp_env_config.json'))
    ap.add_argument('--json', default=None, help='write the flagged list here')
    args = ap.parse_args()

    ref = paper_table(args.pdf)
    cfg = json.load(open(args.config))
    print(f'[audit] {len(ref)} reference rows from {os.path.basename(args.pdf)}\n')

    head = ' '.join(f'{a.upper():>13s}' for a in ALGOS)
    print(f'{"dataset":26s} {head}')
    print(f'{"":26s} ' + ' '.join(f'{"ours / paper":>13s}' for _ in ALGOS))
    flagged, missing = [], []
    for env in ENVS:
        if env not in ref:
            print(f'{env.replace("-v0", ""):26s}  (no reference row)')
            continue
        ours = cfg.get(env, {}).get('ogbench_test_family_mean', {})
        cells = []
        for a in ALGOS:
            r, s = ref[env][a]
            o = ours.get(a)
            if o is None:
                missing.append((env, a))
                cells.append(f'{"--":>6s}/{r:3d}±{s}')
                continue
            o *= 100
            bad = abs(o - r) > max(3 * max(s, 1), 8)
            if bad:
                flagged.append(dict(env=env, algo=a, ours=round(o, 1), paper=r, sd=s))
            cells.append(f'{o:5.1f}{"!" if bad else " "}/{r:3d}±{s}')
        print(f'{env.replace("-v0", ""):26s} ' + ' '.join(f'{c:>13s}' for c in cells))

    print('\n=== outside max(3 sd, 8 points) of the published value ===')
    for f in flagged:
        print(f'  {f["env"].replace("-v0", ""):26s} {f["algo"]:6s} ours {f["ours"]:5.1f} vs paper {f["paper"]} ±{f["sd"]}')
    print(f'flagged {len(flagged)}; bank members absent from the config: {len(missing)}')
    if args.json:
        json.dump(dict(flagged=flagged, missing=missing), open(args.json, 'w'), indent=1)
        print(f'[audit] wrote {args.json}')
    return 1 if flagged else 0


if __name__ == '__main__':
    sys.exit(main())
