import copy
import pytest
import numpy as np
from experience_agent.program import Program, classify
from experience_agent.recovery import Recovery
from experience_agent.memory import Memory
from experience_agent.backends.sonic import dependencies


def prior(task="apple"):
    p = {
        s: {k: [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0] for k in ("left", "right", "torso")}
        for s in ("approach", "contact", "lift")
    }
    return dict(
        task=task,
        poses=p,
        stance=[-0.3, 0.0, 0.0],
        open=[0.0] * 14,
        closed=[1.0] * 14,
        height=0.75,
        active=["right"],
        source_access="keyframes_only",
    )


def ob(**kw):
    x = dict(
        time_s=0.0,
        object_pose=[0.5, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0],
        root_pose=[0.2, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0],
        root_speed_m_s=0.0,
        yaw_speed_rad_s=0.0,
        foot_ground_N=[100.0, 100.0],
        contacts={
            "hand_object_N": 0.0,
            "robot_object_N": 0.0,
            "support_N": 1.0,
            "events": [],
        },
        clearance_m=0.0,
        native_success=False,
    )
    x.update(kw)
    x.update(
        left_pose=x.get("left_pose", [0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]),
        right_pose=x.get("right_pose", [0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]),
    )
    return x


def test_object_motion_retargets_live_approach():
    a = Program(prior())
    a.stage = "approach"
    x = a.step(ob(), 0.02)
    y = a.step(ob(object_pose=[0.5, 0.12, 0.7, 1.0, 0.0, 0.0, 0.0]), 0.02)
    assert y["poses"]["right"][1] - x["poses"]["right"][1] == pytest.approx(0.12)


def test_closure_requires_contact_even_after_elapsed_timeout():
    p = Program(prior())
    p.stage = "contact"
    c = p.step(ob(time_s=20.0), 0.02)
    assert c["fingers"] == [0.0] * 14
    assert p.stage != "lift"
    p = Program(prior())
    p.stage = "contact"
    for _ in range(10):
        c = p.step(
            ob(contacts={"hand_object_N": 2.0, "support_N": 1.0, "events": []}), 0.02
        )
    assert c["fingers"] == [1.0] * 14


def test_lift_and_slip_differ():
    assert (
        classify(
            "apple",
            ob(
                clearance_m=0.12,
                contacts={"hand_object_N": 3.0, "support_N": 0.0},
                native_success=True,
            ),
        )
        == "success"
    )
    assert (
        classify(
            "apple",
            ob(
                clearance_m=0.12,
                contacts={"hand_object_N": 0.0, "support_N": 0.0},
                native_success=True,
            ),
        )
        == "slip"
    )
    assert classify("button", ob(native_success=True)) == "success"
    assert classify("chair", ob(native_success=True)) == "success"


def test_recovery_measured_reason_budget():
    r = Recovery()
    assert r.propose("slip", ob())["change"] == "deeper_contact"
    assert r.propose("contact_timeout", ob())["change"] == "side_approach"
    assert r.propose("slip", ob()) is None
    assert Recovery().propose("nonfinite", ob()) is None


def test_memory_versions_and_no_mock_promotion(tmp_path):
    m = Memory(tmp_path / "memory.json")
    m.record("apple", {"failure": "slip"}, verified=False)
    m.record("apple", {"failure": None}, verified=False)
    assert m.read()["version"] == 2
    assert not m.read()["entries"][-1]["verified_success"]


def test_missing_backend_is_explicit(tmp_path):
    with pytest.raises(RuntimeError, match="runtime dependency"):
        dependencies(tmp_path)


def test_contact_cannot_immediately_lift_before_closed_settles():
    p = Program(prior())
    p.stage = "contact"
    o = ob(
        contacts={
            "hand_object_N": 2.0,
            "robot_object_N": 2.0,
            "support_N": 1.0,
            "events": [],
        }
    )
    for _ in range(5):
        p.step(o, 0.02)
    assert p.stage == "acquire"
    for _ in range(20):
        p.step(o, 0.02)
    assert p.stage == "lift"


def test_recovery_is_distinct_and_direction_uses_measured_error():
    r = Recovery()
    o = ob(right_pose=[0.4, 0.1, 0.8, 1.0, 0.0, 0.0, 0.0])
    a = r.propose("contact_timeout", o)
    b = r.propose("contact_timeout", o)
    assert a["offset_object_m"] != b["offset_object_m"]
    assert a["offset_object_m"][0] > 0 and a["offset_object_m"][1] < 0
    assert (
        Recovery().propose("contact_timeout", o | {"task": "chair"})["change"]
        == "seat_alignment"
    )


def test_controller_snapshot_disambiguates_decoder_and_history():
    from experience_agent.backends.sonic import controller_arrays
    from types import SimpleNamespace

    s = SimpleNamespace(
        last=np.array([2.0]), history=[{"last": np.array([1.0]), "q": np.array([0.0])}]
    )
    z = controller_arrays(s)
    assert z["decoder_last"][0] == 2.0 and z["last"][0, 0] == 1.0


def test_navigation_uses_backend_supported_walk_command_then_brakes():
    p = Program(prior())
    c = p.step(ob(root_pose=[-0.8, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]), 0.02)
    assert np.linalg.norm(c["velocity"]) == pytest.approx(0.5)
    c = p.step(ob(root_pose=[0.17, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]), 0.02)
    assert np.linalg.norm(c["velocity"]) < 0.5


def test_chair_walks_at_measured_standing_height_before_sit():
    x = prior("chair")
    x["height"] = 0.59
    x["seated_height"] = 0.49
    p = Program(x)
    c = p.step(ob(root_pose=[-0.8, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0]), 0.02)
    assert c["height"] == pytest.approx(0.76)
    p.stage = "contact"
    c = p.step(ob(), 0.02)
    assert c["height"] == pytest.approx(0.49)


def test_recovery_action_families_distinguish_contact_slip_and_lift():
    a = Recovery().propose("slip", ob())
    b = Recovery().propose("contact_timeout", ob())
    c = Recovery().propose("lift_timeout", ob())
    assert a["change"] == "deeper_contact" and a["offset_object_m"][2] < 0
    assert b["change"] == "side_approach" and b["offset_object_m"][2] == 0
    assert c["change"] == "higher_proof_lift" and c["offset_object_m"][2] > 0


def test_plate_recovery_changes_failed_left_command_only():
    from experience_agent.program import Program

    x = prior("plate")
    x["active"] = ["left", "right"]
    o = ob(
        task="plate",
        hand_contact_N={"left": 0.0, "right": 3.0},
        hand_target_error_world_m={"left": [0.06, 0.0, 0.0], "right": [0.0, 0.0, 0.0]},
    )
    b = Recovery().propose("contact_timeout", o)
    assert b["active_sides"] == ["left"]
    base = Program(x)
    base.stage = "contact"
    changed = Program(x, b)
    changed.stage = "contact"
    a = base.step(o, 0.02)
    c = changed.step(o, 0.02)
    assert (
        a["poses"]["left"] != c["poses"]["left"]
        and a["poses"]["right"] == c["poses"]["right"]
    )


def test_button_approved_only_native_only_and_hold_loss():
    p = Program(prior("button"))
    p.stage = "contact"
    p.step(ob(native_success=False, approved_success=True), 0.02)
    assert p.stage == "hold"
    p.step(ob(native_success=False, approved_success=False), 0.02)
    assert p.failure == "task_predicate_lost"
    p = Program(prior("button"))
    p.stage = "contact"
    p.step(ob(native_success=True, approved_success=False), 0.02)
    assert p.stage == "contact"


def test_measured_no_acquisition_overrides_lift_timeout_recovery():
    from experience_agent.recovery import recovery_reason

    assert recovery_reason("lift_timeout", "no_acquisition") == "no_acquisition"
    assert recovery_reason("lift_timeout", "insufficient_lift") == "lift_timeout"
