"""External physical runtime boundary. Never imports its replay policy."""

from pathlib import Path
import sys
import copy
import numpy as np


def dependencies(root):
    root = Path(root).resolve()
    for p in (
        root / "humanoid_mimicgen/workflow/native.py",
        root / "src/cap_sim/backends/native_sonic.py",
    ):
        if not p.is_file():
            raise RuntimeError("runtime dependency missing: " + str(p))
    for p in (root, root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return root


class Checkpoint:
    """Full MuJoCo integration, SONIC history, task counters and policy state.

    ONNX inference sessions are stateless and intentionally retained by identity.
    """

    def __init__(self, a, program):
        import mujoco

        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(a.m, self.spec))
        mujoco.mj_getState(a.m, a.d, self.state, self.spec)
        self.sonic = {
            k: copy.deepcopy(v)
            for k, v in vars(a.sonic).items()
            if k not in ("m", "robot", "encoder", "decoder", "planner")
        }
        self.task = {
            k: copy.deepcopy(v)
            for k, v in vars(a.b.e.task).items()
            if k.startswith("_init_")
            or k.endswith("_hold_count")
            or k in ("_contact_started", "reward", "_target_body_name")
        }
        self.target = copy.deepcopy(a.last_target)
        self.program = copy.deepcopy(program)

    def restore(self, a):
        import mujoco

        mujoco.mj_setState(a.m, a.d, self.state, self.spec)
        warm = a.d.qacc_warmstart.copy()
        mujoco.mj_forward(a.m, a.d)
        a.d.qacc_warmstart[:] = warm
        for k, v in self.sonic.items():
            setattr(a.sonic, k, copy.deepcopy(v))
        vars(a.b.e.task).update(copy.deepcopy(self.task))
        a.last_target = copy.deepcopy(self.target)
        check = np.empty_like(self.state)
        mujoco.mj_getState(a.m, a.d, check, self.spec)
        if not np.array_equal(check, self.state):
            raise RuntimeError("checkpoint integration state mismatch")
        return copy.deepcopy(self.program)


def controller_arrays(sonic):
    return dict(
        decoder_last=sonic.last.copy(),
        **{k: np.asarray([h[k] for h in sonic.history]) for k in sonic.history[0]},
    )
