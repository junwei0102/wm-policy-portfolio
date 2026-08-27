"""Snapshot/restore for OGBench environments (locomaze + manipspace).

A snapshot is only valid within a single reset's compiled model: manipspace envs
recompile the MJCF on every reset (`modify_mjcf_model` hook), so branch within an
episode, never across resets.

Restore copies MjData wholesale (all fixed-size array fields + time), NOT
mj_setState + mj_forward. This is deliberate: after mj_step, derived quantities
(site_xpos, xpos, ...) are one substep stale relative to qpos, and the
manipspace action pipeline (`set_control`) reads those stale fields when
converting relative actions to targets. A mj_setState + mj_forward restore
refreshes them, so the replayed branch sees *different* inputs than the
original branch did — ~1e-4 in effector pose, ~1e-2 in observations after one
step. mj_forward also overwrites qacc_warmstart (the solver saves its solution
there on every call), perturbing the next solve at the ULP level. Copying the
data fields verbatim preserves both bitwise.

The size-varying arena views (`contact`, `efc_*`) are skipped: they are
recomputed from scratch inside the next mj_step before anything reads them
across the step boundary. mj_copyData would be the canonical tool but is not
exposed in the mujoco 3.1.6 Python bindings; object references (renderers,
wrappers) also make swapping in a copied MjData unsafe, hence in-place copies.
"""

import copy

import mujoco
import numpy as np

# Python-side attributes that evolve during an episode, per env family. Missing
# attributes are skipped, so one list covers cube/scene/puzzle and maze envs.
_ENV_ATTRS = [
    '_success',
    '_cur_button_states',
    'cur_task_id',
    'cur_task_info',
    'cur_goal_xy',
    '_prev_qpos',
    '_prev_qvel',
    '_prev_ob_info',
]


def _data_array_fields(data):
    for name in dir(data):
        if name.startswith('_') or name == 'contact' or name.startswith('efc_'):
            continue
        try:
            value = getattr(data, name)
        except Exception:
            continue
        if isinstance(value, np.ndarray) and value.size and value.dtype != object:
            yield name, value


class SimBranch:
    """Branch a live OGBench env from mid-episode states."""

    def __init__(self, env):
        self._env = env.unwrapped
        if not hasattr(self._env, 'model') or not hasattr(self._env, 'data'):
            raise ValueError(f'{env} does not expose MuJoCo model/data')
        # OGBench's humanoid steps via the mj_step1/mj_step2 split, so the
        # constraint stage built by the previous step's trailing mj_step1 is
        # live input to the next step. That stage lives in the size-varying
        # arena views we skip, so rebuild it after restore by re-running
        # mj_step1 (a pure function of the restored state). Only do this for
        # split-stepping envs: for manipspace it would refresh the deliberately
        # stale kinematics that set_control reads.
        self._needs_step1 = any(
            '_step_mujoco_simulation' in vars(cls)
            for cls in type(self._env).__mro__
            if 'ogbench' in getattr(cls, '__module__', '')
        )

    @property
    def _data(self):
        return self._env.data

    def snapshot(self):
        arrays = {name: value.copy() for name, value in _data_array_fields(self._data)}
        attrs = {
            name: copy.deepcopy(getattr(self._env, name))
            for name in _ENV_ATTRS
            if hasattr(self._env, name)
        }
        rng = None
        if getattr(self._env, '_np_random', None) is not None:
            rng = copy.deepcopy(self._env._np_random.bit_generator.state)
        return {'arrays': arrays, 'time': self._data.time, 'attrs': attrs, 'rng': rng}

    def restore(self, snap):
        for name, value in snap['attrs'].items():
            setattr(self._env, name, copy.deepcopy(value))
        # Button states also live in mjModel (joint damping locks), so re-apply
        # them before writing mjData.
        if '_cur_button_states' in snap['attrs'] and hasattr(self._env, '_apply_button_states'):
            self._env._apply_button_states()
        for name, saved in snap['arrays'].items():
            current = getattr(self._data, name)
            # Size-varying arena views (constraint islands etc.) can change
            # shape between steps; they are derived state recomputed inside the
            # next mj_step, so skip them rather than copy.
            if current.shape == saved.shape:
                np.copyto(current, saved)
        self._data.time = snap['time']
        if self._needs_step1:
            mujoco.mj_step1(self._env.model, self._data)
        if snap['rng'] is not None:
            self._env._np_random.bit_generator.state = copy.deepcopy(snap['rng'])
