"""Official success predicates on observation pairs, per env family.

The env's success criterion only involves task-relevant dims (cube positions,
agent xy) — never the arm/joint proprio. GCDataset's curgoal labels equate
success with FULL-observation identity, which trains a head that outputs ~0 on
real episode goals (whose arm configuration differs). These predicates apply
the official criterion directly to (obs, goal) vectors so labels match env
semantics.

Verified against real branch outcomes in tests/test_success_predicates.py.

Observation layouts (verified in the task-suite exploration):
- locomaze: obs[0:2] = agent xy (raw units); success = ||xy - goal_xy|| <= 0.5
  (ant/humanoid; pointmaze uses 1.0 but is out of scope).
- manipspace cube-*: 19 proprio dims, then 9 dims per cube
  [(pos - center) * 10 (3), quat (4), cos yaw, sin yaw]. Env success:
  every cube within 0.04 m of its target => 0.4 in obs units.
"""

import numpy as np

_CUBE_COUNTS = {'single': 1, 'double': 2, 'triple': 3, 'quadruple': 4, 'octuple': 8}
_PROPRIO_DIMS = 19
_CUBE_BLOCK = 9
_BUTTON_BLOCK = 4  # [one-hot state (2), joint_pos * 120, joint_vel]
_PUZZLE_BUTTONS = {'3x3': 9, '4x4': 16, '4x5': 20, '4x6': 24}
# Scene (scene_env.py compute_observation): 19 proprio, 1 cube block (9),
# 2 button blocks (4 each), drawer [pos*18, vel], window [pos*15, vel] -> 40.
_SCENE_BUTTONS_OFF = _PROPRIO_DIMS + _CUBE_BLOCK  # 28
_SCENE_DRAWER_OFF = _SCENE_BUTTONS_OFF + 2 * _BUTTON_BLOCK  # 36
_SCENE_WINDOW_OFF = _SCENE_DRAWER_OFF + 2  # 38
# Env success tolerances are 0.04 in raw units; obs scalers convert them.
_SCENE_DRAWER_TOL = 0.04 * 18.0
_SCENE_WINDOW_TOL = 0.04 * 15.0


def _button_states(obs, offset, n_buttons):
    """(B, n_buttons) argmax of each button's one-hot block."""
    idx = offset + _BUTTON_BLOCK * np.arange(n_buttons)
    return (obs[:, idx + 1] > obs[:, idx]).astype(np.int64)


def maze_success(obs, goals, tol=0.5):
    obs, goals = np.atleast_2d(obs), np.atleast_2d(goals)
    return (np.linalg.norm(obs[:, :2] - goals[:, :2], axis=-1) <= tol).astype(np.float32)


def cube_success(num_cubes, obs, goals, tol=0.4):
    obs, goals = np.atleast_2d(obs), np.atleast_2d(goals)
    ok = np.ones(len(obs), dtype=bool)
    for i in range(num_cubes):
        lo = _PROPRIO_DIMS + i * _CUBE_BLOCK
        d = np.linalg.norm(obs[:, lo : lo + 3] - goals[:, lo : lo + 3], axis=-1)
        ok &= d <= tol
    return ok.astype(np.float32)


def scene_success(obs, goals, cube_tol=0.4):
    obs, goals = np.atleast_2d(obs), np.atleast_2d(goals)
    lo = _PROPRIO_DIMS
    cube_ok = np.linalg.norm(obs[:, lo : lo + 3] - goals[:, lo : lo + 3], axis=-1) <= cube_tol
    buttons_ok = (
        _button_states(obs, _SCENE_BUTTONS_OFF, 2) == _button_states(goals, _SCENE_BUTTONS_OFF, 2)
    ).all(axis=-1)
    drawer_ok = np.abs(obs[:, _SCENE_DRAWER_OFF] - goals[:, _SCENE_DRAWER_OFF]) <= _SCENE_DRAWER_TOL
    window_ok = np.abs(obs[:, _SCENE_WINDOW_OFF] - goals[:, _SCENE_WINDOW_OFF]) <= _SCENE_WINDOW_TOL
    return (cube_ok & buttons_ok & drawer_ok & window_ok).astype(np.float32)


def puzzle_success(n_buttons, obs, goals):
    obs, goals = np.atleast_2d(obs), np.atleast_2d(goals)
    return (
        (_button_states(obs, _PROPRIO_DIMS, n_buttons) == _button_states(goals, _PROPRIO_DIMS, n_buttons))
        .all(axis=-1)
        .astype(np.float32)
    )


def scene_progress(obs, goals):
    """Negative sum of tolerance-normalized component distances + button mismatches."""
    obs, goals = np.atleast_2d(obs), np.atleast_2d(goals)
    lo = _PROPRIO_DIMS
    cube = np.linalg.norm(obs[:, lo : lo + 3] - goals[:, lo : lo + 3], axis=-1) / 0.4
    buttons = (
        _button_states(obs, _SCENE_BUTTONS_OFF, 2) != _button_states(goals, _SCENE_BUTTONS_OFF, 2)
    ).sum(axis=-1)
    drawer = np.abs(obs[:, _SCENE_DRAWER_OFF] - goals[:, _SCENE_DRAWER_OFF]) / _SCENE_DRAWER_TOL
    window = np.abs(obs[:, _SCENE_WINDOW_OFF] - goals[:, _SCENE_WINDOW_OFF]) / _SCENE_WINDOW_TOL
    return -(cube + buttons + drawer + window)


def puzzle_progress(n_buttons, obs, goals):
    """Negative Hamming distance between button states, with a continuous
    refinement from the pressed-button joint positions."""
    obs, goals = np.atleast_2d(obs), np.atleast_2d(goals)
    mismatch = (
        _button_states(obs, _PROPRIO_DIMS, n_buttons) != _button_states(goals, _PROPRIO_DIMS, n_buttons)
    ).sum(axis=-1)
    return -mismatch.astype(np.float64)


def maze_progress(obs, goals):
    """Negative xy distance to goal — higher is better."""
    obs, goals = np.atleast_2d(obs), np.atleast_2d(goals)
    return -np.linalg.norm(obs[:, :2] - goals[:, :2], axis=-1)


def cube_progress(num_cubes, obs, goals):
    """Negative mean cube-to-target distance — higher is better."""
    obs, goals = np.atleast_2d(obs), np.atleast_2d(goals)
    total = np.zeros(len(obs))
    for i in range(num_cubes):
        lo = _PROPRIO_DIMS + i * _CUBE_BLOCK
        total += np.linalg.norm(obs[:, lo : lo + 3] - goals[:, lo : lo + 3], axis=-1)
    return -total / num_cubes


def get_progress_fn(env_name):
    """Returns fn(obs (B,d), goals (B,d)) -> (B,) task-relevant progress score."""
    if env_name.startswith(('antmaze', 'humanoidmaze')):
        return maze_progress
    if env_name.startswith('cube'):
        num = _CUBE_COUNTS[env_name.split('-')[1]]
        return lambda obs, goals: cube_progress(num, obs, goals)
    if env_name.startswith('scene'):
        return scene_progress
    if env_name.startswith('puzzle'):
        num = _PUZZLE_BUTTONS[env_name.split('-')[1]]
        return lambda obs, goals: puzzle_progress(num, obs, goals)
    raise NotImplementedError(f'no progress fn for {env_name} yet')


def get_predicate(env_name):
    """Returns fn(obs (B,d), goals (B,d)) -> (B,) float {0,1}."""
    if env_name.startswith(('antmaze', 'humanoidmaze')):
        return lambda obs, goals: maze_success(obs, goals, tol=0.5)
    if env_name.startswith('cube'):
        num = _CUBE_COUNTS[env_name.split('-')[1]]
        return lambda obs, goals: cube_success(num, obs, goals)
    if env_name.startswith('scene'):
        return scene_success
    if env_name.startswith('puzzle'):
        num = _PUZZLE_BUTTONS[env_name.split('-')[1]]
        return lambda obs, goals: puzzle_success(num, obs, goals)
    raise NotImplementedError(f'no success predicate for {env_name} yet')
