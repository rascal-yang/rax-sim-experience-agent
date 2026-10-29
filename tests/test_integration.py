import numpy as np
import pytest
from experience_agent.planning import slew
from experience_agent.priors import extract
from experience_agent.runner import audit_actions


def test_slew_bounds_and_preserves_unit_orientation():
    a = {"right": np.array([0, 0, 0, 1.0, 0, 0, 0])}
    b = {"right": np.array([1.0, 0, 0, 0, 0, 0, 1.0])}
    c = slew(a, b, 0.02)
    assert np.linalg.norm(c["right"][:3] - a["right"][:3]) <= 0.00401
    assert np.linalg.norm(c["right"][3:]) == pytest.approx(1.0)


def test_action_audit_catches_missing_hand_mapping():
    with pytest.raises(ValueError, match="coverage"):
        audit_actions([{"mode": 1, "token": [1] * 64}], [])
    assert audit_actions(
        [{"mode": 1, "token": [1] * 64, "time": 0.0}],
        [{"token_index": 0, "fingers": [0] * 14, "time_s": 0.0}],
    )["passed"]


def test_real_source_extraction_no_runtime_sequence_access():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "cap-sim-api-augmentation"
    p = extract(root, "apple")
    assert p["source_access"] == "keyframes_only"
    assert set(p["poses"]) == {"approach", "contact", "lift"}
    assert "qpos" not in p and "source_rows" not in p
    assert p["provenance"]["original_actions_visible_during_extraction"]


@pytest.mark.parametrize(
    "task,obj,active",
    [
        ("apple", "apple_001", ["right"]),
        ("plate", "plate_3", ["left", "right"]),
        ("button", "button002_base", ["right"]),
        ("chair", "chair_319_0p9", []),
    ],
)
def test_exact_task_bindings(task, obj, active):
    from pathlib import Path

    p = extract(Path(__file__).resolve().parents[2] / "cap-sim-api-augmentation", task)
    assert p["bindings"]["object"] == obj
    assert p["active"] == active
    assert p["acceptance"] == ("button_palm_mesh_v1" if task == "button" else "native")


def test_action_audit_rejects_wrong_temporal_mapping_and_nonfinite_hands():
    records = [
        {"mode": 1, "token": [1.0] * 64, "time": 0.0},
        {"mode": 1, "token": [2.0] * 64, "time": 0.02},
    ]
    commands = [
        {"time_s": i * 0.005, "token_index": i // 4, "fingers": [0.0] * 14}
        for i in range(8)
    ]
    assert audit_actions(records, commands)["passed"]
    import copy

    wrong = copy.deepcopy(commands)
    wrong[-1]["token_index"] = 0
    with pytest.raises(ValueError, match="temporal"):
        audit_actions(records, wrong)
    wrong = copy.deepcopy(commands)
    wrong[2]["fingers"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        audit_actions(records, wrong)
    with pytest.raises(ValueError, match="unreferenced"):
        audit_actions(records, commands[:4])


def test_execution_error_preserves_attempt_and_actual_integrated_count(tmp_path):
    import json
    from types import SimpleNamespace
    from experience_agent.runner import run
    from experience_agent.contracts import sha
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_program import prior, ob

    p = prior()
    p["acceptance"] = "native"
    data = SimpleNamespace(
        time=0.0,
        qpos=np.array([-0.8, 0, 0.75, 1, 0, 0, 0.0]),
        qvel=np.zeros(6),
        ctrl=np.zeros(1),
        warning=[],
    )
    a = SimpleNamespace(
        d=data,
        dt=0.005,
        control_stride=4,
        sonic=SimpleNamespace(records=[]),
        last_target=np.zeros(29),
    )
    a.poses = lambda: {
        s: np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
        for s in ("left", "right", "torso")
    }
    a.observe = lambda: ob(time_s=data.time, root_pose=data.qpos.tolist())
    a.close = lambda: None
    a.native_success = lambda: False

    def command(*args, **kwargs):
        if data.time > 0:
            raise RuntimeError("injected actuator failure")
        a.sonic.records.append({"time": data.time, "mode": 1, "token": [1.0] * 64})

    a.command = command

    def step():
        data.time += 0.005

    a.step = step
    (tmp_path / "experiment.json").write_text(
        json.dumps({"preflight": {"evidence_sha256": {}}})
    )
    (tmp_path / "experiment.sha256").write_text(sha(tmp_path / "experiment.json"))
    with pytest.raises(RuntimeError, match="injected actuator"):
        run(tmp_path, a, p)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["attempts"][0]["actual_integrated_steps_all_branches"] == 1
    assert result["actual_integrated_steps_all_branches"] == 1
    assert result["failure"].startswith("execution_error")


def test_inherited_checker_and_initializer_are_frozen(tmp_path):
    from experience_agent.experiment import dependency_sources, verify_sources
    from experience_agent.contracts import sha
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    files = dependency_sources(
        root / "cap-sim-experience-agent", root / "cap-sim-api-augmentation", None
    )
    assert (
        root / "cap-sim-api-augmentation/humanoid_mimicgen/augmentation/direct_grasp.py"
        in files
    )
    assert root / "cap-sim/scripts/humano_python.sh" in files
    p = tmp_path / "inherited_checker.py"
    p.write_text("x=1")
    evidence = {str(p): sha(p)}
    p.write_text("x=2")
    with pytest.raises(ValueError, match="Frozen evidence changed"):
        verify_sources(evidence)


def test_full_audit_rejects_deleted_state_and_wrong_token_dimension():
    records = [{"time": 0.0, "mode": 1, "token": [0.0] * 64, "target": [0.0] * 29}]
    commands = [
        {
            "time_s": i * 0.005,
            "token_index": 0,
            "fingers": [0.0] * 14,
            "body_target_rad": [0.0] * 29,
        }
        for i in range(4)
    ]
    state = np.zeros((4, 14))
    state[:, 0] = np.arange(4) * 0.005
    next_state = state.copy()
    next_state[:, 0] += 0.005
    trace = {
        "state": state,
        "next_state": next_state,
        "commands": np.c_[state[:, 0], np.zeros((4, 30))],
    }
    assert audit_actions(records, commands, trace)["trajectory_coverage_checked"]
    with pytest.raises(ValueError, match="length"):
        audit_actions(records, commands, trace | {"state": state[:-1]})
    records[0]["token"] = [0.0] * 32
    with pytest.raises(ValueError, match="dimension"):
        audit_actions(records, commands, trace)


def test_navigation_root_attached_reference_has_no_world_speed_lag():
    from experience_agent.planning import navigation_poses

    initial = {"torso": np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])}
    moved = navigation_poses(
        initial, np.array([0.0, 0.0, 0.7]), np.array([0.2, 0.0, 0.7])
    )
    assert moved["torso"][0] == pytest.approx(0.2)


def test_successful_task_with_rejected_audit_cannot_be_verified(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import experience_agent.runner as runner
    from experience_agent.contracts import sha
    from experience_agent.memory import Memory
    from test_program import prior, ob

    class CompletedProgram:
        stage = "hold"
        button_acceptance = None
        transitions = []

        def __init__(self, p):
            self.calls = 0

        def step(self, o, dt):
            self.calls += 1
            return dict(
                poses={
                    s: [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
                    for s in ("left", "right", "torso")
                },
                velocity=[0.0, 0.0],
                facing=0.0,
                height=0.75,
                fingers=[0.0] * 14,
                failure=None,
                completed=self.calls > 1,
            )

    monkeypatch.setattr(runner, "Program", CompletedProgram)
    data = SimpleNamespace(
        time=0.0,
        qpos=np.array([0.0, 0, 0.75, 1, 0, 0, 0.0]),
        qvel=np.zeros(6),
        ctrl=np.zeros(1),
        warning=[],
    )
    a = SimpleNamespace(
        d=data,
        dt=0.005,
        control_stride=4,
        sonic=SimpleNamespace(records=[]),
        last_target=np.zeros(29),
    )
    a.poses = lambda: {
        s: np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
        for s in ("left", "right", "torso")
    }
    a.observe = lambda: ob(
        time_s=data.time,
        native_success=True,
        clearance_m=0.12,
        contacts={"hand_object_N": 2.0, "support_N": 0.0, "events": []},
    )
    a.close = lambda: None
    a.native_success = lambda: True

    def command(*args, policy_tick):
        if policy_tick:
            a.sonic.records.append(
                {
                    "time": data.time,
                    "mode": 1,
                    "token": [1.0] * 32,
                    "target": [0.0] * 29,
                }
            )

    a.command = command

    def step():
        data.time += 0.005

    a.step = step
    (tmp_path / "experiment.json").write_text(
        json.dumps({"preflight": {"evidence_sha256": {}}})
    )
    (tmp_path / "experiment.sha256").write_text(sha(tmp_path / "experiment.json"))
    p = prior()
    p["acceptance"] = "native"
    result = runner.run(tmp_path, a, p)
    assert result["task_outcome_success"] and not result["success"]
    assert result["verification_failure"] and not result["record_valid"]
    assert (
        json.loads((tmp_path / "execution_audit.json").read_text())["success"] is False
    )
    m = Memory(tmp_path / "memory.json")
    m.record("apple", result, verified=True)
    assert not m.read()["entries"][-1]["verified_success"]


def test_real_grasp_priors_have_open_contact_and_upward_proof_evidence():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "cap-sim-api-augmentation"
    for task in ("apple", "plate"):
        p = extract(root, task)
        assert not np.allclose(p["open"], p["closed"])
        e = p["extraction_evidence"]
        assert e["phases"]["lift"]["native_success"]
        assert (
            e["phases"]["lift"]["object_xyz"][2]
            > e["phases"]["contact"]["object_xyz"][2]
        )
        assert e["hand_alignment"] == "actual controller timestamps; sample-and-hold"
    assert p["source_access"] == "keyframes_only"
