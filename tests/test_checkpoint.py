"""Serialization continuity test only: no mj_step or physical success claim."""

from types import SimpleNamespace
from collections import deque
import numpy as np
import mujoco
from experience_agent.backends.sonic import Checkpoint


def test_complete_checkpoint_restores_integrator_controller_task_and_program():
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    s = SimpleNamespace(
        m=m,
        robot=object(),
        encoder=object(),
        decoder=object(),
        planner=object(),
        history=deque([np.ones(3)], maxlen=10),
        last=np.array([2.0]),
        tick=11,
        records=[{"mode": 1}],
        ref=np.ones((10, 36)),
        serial=3,
    )
    task = SimpleNamespace(_hold_count=7, _init_x=np.array([1.0]))
    a = SimpleNamespace(
        m=m,
        d=d,
        sonic=s,
        b=SimpleNamespace(e=SimpleNamespace(task=task)),
        last_target=np.array([3.0]),
    )
    d.qacc_warmstart[:] = np.arange(m.nv) + 1.0
    cp = Checkpoint(a, {"stage": "approach", "elapsed": 0.2})
    before = d.qpos.copy()
    d.qpos[0] = 4.0
    s.last[:] = 100
    s.history.append(np.zeros(3))
    task._hold_count = 0
    a.last_target[:] = 0
    program = cp.restore(a)
    assert np.array_equal(d.qpos, before)
    assert s.last[0] == 2 and len(s.history) == 1 and task._hold_count == 7
    assert a.last_target[0] == 3 and program["stage"] == "approach"


def test_checkpoint_retains_runtime_lock_by_identity():
    import threading

    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    lock = threading.Lock()
    task = SimpleNamespace(runtime_resource=lock, _button_hold_count=5)
    a = SimpleNamespace(
        m=m,
        d=d,
        sonic=SimpleNamespace(
            m=m,
            robot=object(),
            encoder=object(),
            decoder=object(),
            planner=object(),
            tick=0,
            history=[],
        ),
        b=SimpleNamespace(e=SimpleNamespace(task=task)),
        last_target=None,
    )
    cp = Checkpoint(a, {"stage": "approach"})
    task._button_hold_count = 0
    cp.restore(a)
    assert task.runtime_resource is lock and task._button_hold_count == 5
