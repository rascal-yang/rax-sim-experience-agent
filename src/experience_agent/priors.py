"""Offline extraction of task phases from verified source contact/outcome evidence.

The extraction boundary reads source states, actual controller timestamps,
commands and measured outcomes. Execution sees spatial relations and exemplars
only, with phase provenance; it never samples the source action sequence.
"""

import json
from pathlib import Path
import numpy as np
from .contracts import sha
from .backends.sonic import dependencies


def extract(runtime, task, manifest=None):
    runtime = dependencies(runtime)
    import mujoco

    manifest = (
        Path(manifest)
        if manifest
        else runtime
        / "humanoid_mimicgen/workflow/tests/fixtures/navigation"
        / ("apple_lateral.json" if task == "apple" else task + ".json")
    )
    source = json.loads(manifest.read_text())
    files = source["files"]
    baseline = Path(files["model"]).parent
    result = json.loads(Path(files["result"]).read_text())
    accepted = result.get(source["success_field"])
    if task == "button":
        accepted = json.loads(Path(files["acceptance"]).read_text())["revised_success"]
    if not accepted:
        raise ValueError("Source lacks task-specific verified success")
    model = mujoco.MjModel.from_binary_path(files["model"])
    data = mujoco.MjData(model)
    obj = model.body(source["bindings"]["object"]).id
    with np.load(files["trajectory"]) as z:
        states = z[source["state_key"]].copy()
    with np.load(baseline / "execution/controller.npz") as z:
        controller_times = z["records"][:, 0].copy()
    with np.load(baseline / "execution/hand_commands.npz") as z:
        hands = np.concatenate([z["left"], z["right"]], axis=1)
    if len(hands) != len(controller_times) or np.any(np.diff(controller_times) <= 0):
        raise ValueError("Source controller/hand cadence mismatch")
    metrics = json.loads((baseline / "execution/metrics.json").read_text())
    times = states[:, 0]
    first, last = source["entry_index"], source["end_index"]
    available = [
        (i, m)
        for i, m in enumerate(metrics)
        if times[first] - 1e-8 <= m["time_s"] <= times[last] + 0.021
    ]
    if not available:
        raise ValueError("No measured source phase outcomes")

    def row_at(time):
        return int(np.argmin(abs(times - time)))

    def object_xyz(metric):
        return np.asarray(next(iter(metric["object_xyz"].values())), float)

    if task in ("apple", "plate"):
        contacts = [(i, m) for i, m in available if m["hand_object_N"] > 0]
        successes = [
            (i, m)
            for i, m in available
            if m["native_success"] and m["hand_object_N"] > 0
        ]
        if not contacts or not successes:
            raise ValueError("Verified grasp lacks measured contact/success phase")
        supported = [
            (i, m)
            for i, m in contacts
            if m["support_N"] > 0 and m["time_s"] < successes[0][1]["time_s"]
        ]
        if not supported:
            raise ValueError("Grasp source lacks supported acquisition boundary")
        contact_metric = supported[-1][1]
        lift_metric = max(successes, key=lambda x: object_xyz(x[1])[2])[1]
    elif task == "chair":
        contacts = [
            (i, m)
            for i, m in available
            if m.get("_sit_metrics", {}).get("has_chair_contact")
        ]
        successes = [(i, m) for i, m in available if m["native_success"]]
        if not contacts or not successes:
            raise ValueError("Chair source lacks native seated-contact evidence")
        contact_metric = contacts[0][1]
        lift_metric = successes[0][1]
    else:
        # Approved palm source can have native_success=False. Select its measured
        # strongest selected contact, not an early finger change or terminal row.
        contact_metric = max(
            available, key=lambda x: x[1]["contact_control"]["selected_force_N"]
        )[1]
        lift_metric = contact_metric
        if contact_metric["contact_control"]["selected_force_N"] < 20:
            raise ValueError("Approved button source lacks20N contact evidence")
    indices = {
        "approach": first,
        "contact": row_at(contact_metric["time_s"]),
        "lift": row_at(lift_metric["time_s"]),
    }
    body_names = {
        "left": "left_wrist_yaw_link",
        "right": "right_wrist_yaw_link",
        "torso": "torso_link",
    }
    poses = {}
    objects = {}
    qposes = {}
    evidence = {}
    metric_for = {"contact": contact_metric, "lift": lift_metric}
    for phase, index in indices.items():
        q = states[index, 1 : 1 + model.nq]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        anchor = data.xpos[obj].copy()
        objects[phase] = anchor
        qposes[phase] = q.copy()
        poses[phase] = {}
        for side, name in body_names.items():
            bid = model.body(name).id
            poses[phase][side] = np.r_[
                data.xpos[bid] - anchor, data.xquat[bid]
            ].tolist()
        hand_index = int(
            np.searchsorted(controller_times, times[index] + 1e-8, side="right") - 1
        )
        if hand_index < 0 or hand_index >= len(hands):
            raise ValueError("Source hand time outside controller coverage")
        measured = metric_for.get(phase, {})
        evidence[phase] = dict(
            source_row=index,
            state_time_s=float(times[index]),
            controller_index=hand_index,
            controller_time_s=float(controller_times[hand_index]),
            object_xyz=anchor.tolist(),
            hand_object_N=measured.get("hand_object_N"),
            robot_object_N=measured.get("robot_object_N"),
            support_N=measured.get("support_N"),
            native_success=measured.get("native_success"),
            selected_contact_force_N=measured.get("contact_control", {}).get(
                "selected_force_N"
            ),
        )
    # Proof target is the demonstrated peak relative to the acquisition object,
    # rather than relative to the already lifted object or a late lowered state.
    for side in poses["lift"]:
        poses["lift"][side][:3] = (
            np.asarray(poses["lift"][side][:3]) + objects["lift"] - objects["contact"]
        ).tolist()
    active = (
        ["left", "right"] if task == "plate" else [] if task == "chair" else ["right"]
    )
    open_hand = hands[evidence["approach"]["controller_index"]]
    closed_hand = hands[evidence["lift"]["controller_index"]]
    if task in ("apple", "plate"):
        if np.allclose(open_hand, closed_hand):
            raise ValueError(
                "Grasp source entry already closed; missing open-hand phase"
            )
        if objects["lift"][2] <= objects["contact"][2]:
            raise ValueError("Grasp proof object height is not above acquisition")
    stance = np.r_[qposes["approach"][:2] - objects["approach"][:2], 0.0].tolist()
    paths = [
        manifest,
        Path(files["model"]),
        Path(files["trajectory"]),
        Path(files["result"]),
        baseline / "execution/hand_commands.npz",
        baseline / "execution/controller.npz",
        baseline / "execution/metrics.json",
    ]
    button_profile = {}
    if task == "button":
        base = json.loads((baseline / "experiment.json").read_text())
        button_profile = dict(
            button_limits=base["physics"]["safety_gates"],
            button_selected_names=json.loads(
                Path(files["contact_contract"]).read_text()
            )["contact_control"]["contact_bodies"],
        )
        paths.append(Path(files["acceptance"]))
    return dict(
        **button_profile,
        schema_version=2,
        task=task,
        poses=poses,
        stance=stance,
        height=float(qposes["approach"][2]),
        seated_height=float(qposes["lift"][2]),
        open=open_hand.tolist(),
        closed=closed_hand.tolist(),
        contact_fingers=hands[evidence["contact"]["controller_index"]].tolist(),
        active=active,
        source_access="keyframes_only",
        extraction_evidence=dict(
            source_manifest=str(manifest),
            source_manifest_sha256=sha(manifest),
            source_interval=[first, last],
            hand_alignment="actual controller timestamps; sample-and-hold",
            phase_selection="last supported measured hand contact before native success, and native-success object-height peak for grasps; native seat contact for chair; approved source peak selected force for button",
            phases=evidence,
            proof_object_displacement_m=(objects["lift"] - objects["contact"]).tolist(),
            proof_wrist_displacement_world_m={
                side: (
                    np.asarray(poses["lift"][side][:3])
                    - np.asarray(poses["contact"][side][:3])
                ).tolist()
                for side in active
            },
        ),
        provenance=dict(
            original_actions_visible_during_extraction=True,
            original_actions_visible_during_execution=False,
            source_sha256={str(p): sha(p) for p in paths},
            reference_baseline="historical trajectory retargeting; not a paired controller comparison",
        ),
        bindings=source["bindings"],
        acceptance=source["success_criterion"],
    )
