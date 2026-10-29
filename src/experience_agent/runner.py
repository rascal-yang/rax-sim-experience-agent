"""Own execution loop, checkpoint branches and explicit token/hand audit."""

import copy
import json
import time
from pathlib import Path
import numpy as np
from .contracts import now, save, sha
from .program import Program, classify
from .planning import slew, navigation_poses
from .recovery import Recovery, recovery_reason
from .memory import Memory
from .backends.sonic import dependencies, Checkpoint


def audit_actions(records, commands, trace=None):
    if not records or not commands:
        raise ValueError("token/hand coverage missing")
    if any(np.asarray(r["token"]).shape != (64,) for r in records):
        raise ValueError("SONIC token dimension must be64")
    if trace is not None:
        if any(
            len(trace[k]) != len(commands) for k in ("state", "next_state", "commands")
        ):
            raise ValueError("trajectory length coverage mismatch")
        state = np.asarray(trace["state"])
        nxt = np.asarray(trace["next_state"])
        control = np.asarray(trace["commands"])
        if (
            state.shape != nxt.shape
            or state.ndim != 2
            or control.ndim != 2
            or not np.isfinite(state).all()
            or not np.isfinite(nxt).all()
            or not np.isfinite(control).all()
        ):
            raise ValueError("trajectory finite shape violation")
        expected_times = np.asarray([c["time_s"] for c in commands])
        if (
            not np.allclose(state[:, 0], expected_times, atol=1e-8, rtol=0)
            or not np.allclose(nxt[:, 0], expected_times + 0.005, atol=1e-8, rtol=0)
            or not np.allclose(control[:, 0], expected_times, atol=1e-8, rtol=0)
        ):
            raise ValueError("trajectory timestamp coverage mismatch")
        if len(state) > 1 and not np.array_equal(state[1:], nxt[:-1]):
            raise ValueError("trajectory checkpoint continuity violation")
    times = np.asarray([r.get("time", float("nan")) for r in records])
    steps = np.asarray([c.get("time_s", float("nan")) for c in commands])
    if not np.isfinite(times).all() or not np.isfinite(steps).all():
        raise ValueError("temporal timestamps missing")
    if len(steps) > 1 and not np.allclose(np.diff(steps), 0.005, atol=1e-8, rtol=0):
        raise ValueError("temporal physics cadence mismatch")
    if len(times) > 1 and not np.allclose(np.diff(times), 0.02, atol=1e-8, rtol=0):
        raise ValueError("temporal policy cadence mismatch")
    seen = set()
    for c in commands:
        index = c["token_index"]
        if not 0 <= index < len(records) or len(c["fingers"]) != 14:
            raise ValueError("token/hand coverage mismatch")
        if not np.isfinite(c["fingers"]).all():
            raise ValueError("hands must be finite")
        expected = int(np.searchsorted(times, c["time_s"] + 1e-8, side="right") - 1)
        if index != expected or not -1e-8 <= c["time_s"] - times[index] < 0.02 - 1e-8:
            raise ValueError("temporal token mapping mismatch")
        if (
            "target" in records[index]
            and "body_target_rad" in c
            and not np.array_equal(records[index]["target"], c["body_target_rad"])
        ):
            raise ValueError("postdecoder target differs from recorded SONIC output")
        seen.add(index)
    if len(seen) != len(records):
        raise ValueError("unreferenced body tokens")
    if any(r["mode"] != 1 or not np.isfinite(r["token"]).all() for r in records):
        raise ValueError("SONIC actual mode/token violation")
    return dict(
        passed=True,
        actual_modes=[1],
        original_token_steps=sum(r["mode"] == 0 for r in records),
        trajectory_coverage_checked=trace is not None,
        body_tokens=len(records),
        hand_steps=len(commands),
        post_decoder_override=(
            False
            if all(
                "target" in records[c["token_index"]] and "body_target_rad" in c
                for c in commands
            )
            else None
        ),
    )


def run(case, a, prior, budget=80.0):
    import mujoco
    from .backends.sonic import Checkpoint

    case = Path(case)
    contract = json.loads((case / "experiment.json").read_text())
    if (
        sha(case / "experiment.json")
        != (case / "experiment.sha256").read_text().strip()
    ):
        raise ValueError("Frozen contract changed")
    for p, h in contract["preflight"]["evidence_sha256"].items():
        if sha(p) != h:
            raise ValueError("Frozen evidence changed: " + p)
    program = Program(prior)
    poses = a.poses()
    initial = copy.deepcopy(poses)
    root = a.d.qpos[:3].copy()
    recovery = Recovery()
    attempts = []
    attempt_started = now()
    attempt_step_start = 0
    inherited_prefix_steps = 0
    checkpoint = None
    started = now()
    clock = time.monotonic()
    planning = 0.0
    physics = 0.0
    actual_steps = 0
    states = []
    nexts = []
    commands = []
    ctrls = []
    observations = []
    ever = False
    native = False
    failure = None
    prefix = None
    branch = 0
    finished = False
    resuming = False
    try:
        for k in range(round(budget / a.dt) * 3):
            tick = k % a.control_stride == 0
            if resuming and not tick:
                continue
            resuming = False
            if tick:
                ob = a.observe()
                ob["native_success"] = native
                ob["task"] = prior["task"]
                if prior["task"] == "button":
                    from .acceptance import button_contact

                    ob["button_contact"] = button_contact(
                        a, ob, prior["button_selected_names"]
                    )
                ob["hand_target_error_world_m"] = {
                    side: (
                        np.asarray(prior["poses"]["contact"][side][:3])
                        + np.asarray(ob["object_pose"][:3])
                        - np.asarray(ob[side + "_pose"][:3])
                    ).tolist()
                    for side in prior["active"]
                }
                if hasattr(a, "hands"):
                    ob["hand_contact_N"] = {
                        side: sum(
                            event["force_N"]
                            for event in ob["contacts"]["events"]
                            if any(
                                body in {a.m.body(i).name for i in group}
                                for body in event["bodies"]
                            )
                            and any(
                                body in {a.m.body(i).name for i in a.object_bodies}
                                for body in event["bodies"]
                            )
                        )
                        for side, group in a.hands.items()
                    }
                t = time.monotonic()
                cmd = program.step(ob, 0.02)
                planning += time.monotonic() - t
                if program.stage == "navigation":
                    desired = navigation_poses(initial, root, a.d.qpos[:3])
                    poses = desired
                else:
                    desired = cmd["poses"]
                    poses = slew(poses, desired, 0.02)
                if checkpoint is None and program.stage == "approach":
                    checkpoint = Checkpoint(a, program)
                    import gzip, pickle

                    with gzip.open(case / "complete_checkpoint.pkl.gz", "wb") as f:
                        pickle.dump(checkpoint, f)
                    prefix = copy.deepcopy(
                        dict(
                            states=states,
                            nexts=nexts,
                            commands=commands,
                            ctrls=ctrls,
                            observations=observations,
                            poses=poses,
                            native=native,
                            ever=ever,
                        )
                    )
                if (
                    cmd["failure"]
                    or cmd["completed"]
                    or len(commands) * a.dt >= budget * (branch + 1)
                ):
                    failure = cmd["failure"] or (
                        None if cmd["completed"] else "simulation_budget"
                    )
                    attempts.append(
                        dict(
                            branch=branch,
                            failure=failure,
                            outcome=classify(prior["task"], ob),
                            native_ever=ever,
                            native_terminal=native,
                            started_at=attempt_started,
                            inherited_prefix_steps=inherited_prefix_steps,
                            newly_integrated_steps=actual_steps - attempt_step_start,
                            executed_steps=len(commands),
                            ended_at=now(),
                        )
                    )
                    proposal = (
                        recovery.propose(
                            recovery_reason(
                                failure or classify(prior["task"], ob),
                                classify(prior["task"], ob),
                            ),
                            ob,
                        )
                        if failure and checkpoint
                        else None
                    )
                    if proposal:
                        branch_dir = case / "branches" / str(branch)
                        branch_dir.mkdir(parents=True)
                        save(branch_dir / "commands.json", commands)
                        save(branch_dir / "observations.json", observations)
                        save(branch_dir / "tokens.json", a.sonic.records)
                        save(branch_dir / "proposal.json", proposal)
                        np.savez_compressed(
                            branch_dir / "trajectory.npz",
                            state=states,
                            next_state=nexts,
                            commands=ctrls,
                        )
                        program = checkpoint.restore(a)
                        program.branch = proposal
                        program.elapsed = 0.0
                        program.failure = None
                        states = copy.deepcopy(prefix["states"])
                        nexts = copy.deepcopy(prefix["nexts"])
                        commands = copy.deepcopy(prefix["commands"])
                        ctrls = copy.deepcopy(prefix["ctrls"])
                        observations = copy.deepcopy(prefix["observations"])
                        poses = copy.deepcopy(prefix["poses"])
                        native = prefix["native"]
                        ever = prefix["ever"]
                        branch += 1
                        resuming = True
                        attempt_started = now()
                        attempt_step_start = actual_steps
                        inherited_prefix_steps = len(commands)
                        save(
                            case / ("checkpoint_restore_" + str(branch) + ".json"),
                            dict(
                                integration_exact=True,
                                sonic_tick=a.sonic.tick,
                                history_size=len(a.sonic.history),
                                prefix_steps=len(commands),
                                time_s=a.d.time,
                            ),
                        )
                        continue
                    finished = bool(cmd["completed"])
                    break
            a.command(
                poses,
                np.asarray(cmd["velocity"]),
                cmd["facing"],
                cmd["height"],
                np.asarray(cmd["fingers"]),
                policy_tick=tick,
            )
            states.append(np.r_[a.d.time, a.d.qpos.copy(), a.d.qvel.copy()])
            ctrls.append(np.r_[a.d.time, a.last_target, a.d.ctrl.copy()])
            commands.append(
                dict(
                    time_s=float(a.d.time),
                    stage=program.stage,
                    token_index=len(a.sonic.records) - 1,
                    fingers=cmd["fingers"],
                    body_target_rad=a.last_target.tolist(),
                    desired=copy.deepcopy(cmd["poses"]),
                    encoder_pose={s: v.tolist() for s, v in poses.items()},
                    branch=branch,
                )
            )
            t = time.monotonic()
            a.step()
            physics += time.monotonic() - t
            actual_steps += 1
            if not np.isfinite(a.d.qpos).all():
                raise RuntimeError("nonfinite simulation")
            nexts.append(np.r_[a.d.time, a.d.qpos.copy(), a.d.qvel.copy()])
            o = a.observe()
            if (len(commands) % a.control_stride) == 0:
                native = a.native_success()
                ever = ever or native
            o.update(stage=program.stage, native_success=native, branch=branch)
            if program.button_acceptance is not None:
                o["approved_success"] = program.button_acceptance.success
            observations.append(o)
    except BaseException as exc:
        failure = (
            "interrupted"
            if isinstance(exc, KeyboardInterrupt)
            else "execution_error:" + type(exc).__name__ + ":" + str(exc)
        )
        attempts.append(
            dict(
                branch=branch,
                failure=failure,
                outcome="execution_error",
                native_ever=ever,
                native_terminal=native,
                started_at=attempt_started,
                inherited_prefix_steps=inherited_prefix_steps,
                newly_integrated_steps=actual_steps - attempt_step_start,
                ended_at=now(),
                executed_steps=len(commands),
                actual_integrated_steps_all_branches=actual_steps,
            )
        )
        raise
    finally:
        try:
            out = case / "execution"
            out.mkdir(exist_ok=True)
            np.savez_compressed(
                out / "trajectory.npz", state=states, next_state=nexts, commands=ctrls
            )
            save(out / "commands.json", commands)
            save(out / "observations.json", observations)
            save(out / "controller.json", a.sonic.records)
            if commands:
                np.savez_compressed(
                    out / "sonic_actions.npz",
                    tokens=np.asarray([r["token"] for r in a.sonic.records]),
                    hands=np.asarray([c["fingers"] for c in commands]),
                    step_token=np.asarray([c["token_index"] for c in commands]),
                )
            try:
                audit = (
                    audit_actions(
                        a.sonic.records,
                        commands,
                        dict(state=states, next_state=nexts, commands=ctrls),
                    )
                    if commands
                    else dict(passed=False)
                )
            except ValueError as exc:
                audit = dict(passed=False, error=str(exc))
            audit["simulator_warning_count"] = int(sum(w.number for w in a.d.warning))
            audit["recorded_state_continuity"] = bool(
                len(states) < 2
                or np.array_equal(np.asarray(states)[1:], np.asarray(nexts)[:-1])
            )
            task_outcome_success = bool(
                finished
                and observations
                and classify(prior["task"], observations[-1]) == "success"
            )
            record_valid = bool(
                audit.get("passed") and audit.get("recorded_state_continuity")
            )
            result = dict(
                experiment_id=case.name,
                started_at=started,
                ended_at=now(),
                task_outcome_success=task_outcome_success,
                record_valid=record_valid,
                verification_failure=(
                    None
                    if record_valid
                    else audit.get("error", "execution evidence failed audit")
                ),
                physical_evidence_kind=(
                    "native_mujoco"
                    if type(a).__module__ == "humanoid_mimicgen.workflow.native"
                    and type(a).__name__ == "NativeAdapter"
                    else "nonphysical_test"
                ),
                success=bool(task_outcome_success and record_valid and failure is None),
                native_success=ever,
                native_terminal=native,
                completed=finished,
                failure=failure,
                attempts=attempts,
                transitions=program.transitions,
                audit=audit,
                physics_steps=len(commands),
                actual_integrated_steps_all_branches=actual_steps,
                stage_coverage={
                    s: sum(c["stage"] == s for c in commands)
                    for s in (
                        "navigation",
                        "settle",
                        "approach",
                        "contact",
                        "acquire",
                        "lift",
                        "hold",
                        "done",
                    )
                },
                timing=dict(
                    planning_wall_s=planning,
                    physics_step_compute_wall_s=physics,
                    total_execution_wall_s=time.monotonic() - clock,
                    simulation_s=float(a.d.time),
                    rendering_wall_s=None,
                ),
                task=prior["task"],
                acceptance=prior["acceptance"],
                retained_grasp=(
                    bool(observations)
                    and classify(prior["task"], observations[-1]) == "success"
                    if prior["task"] in ("apple", "plate")
                    else None
                ),
            )
            save(case / "result.json", result)
            save(case / "execution_audit.json", result)
        finally:
            a.close()
    return result
