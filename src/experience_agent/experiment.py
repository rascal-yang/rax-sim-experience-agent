"""Complete prephysics freeze from the required parent template."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from .contracts import now, sha, save
from .backends.sonic import dependencies, controller_arrays
from .priors import extract


def prepare(repo, runtime, task, offset):
    runtime = dependencies(runtime)
    repo = Path(repo).resolve()
    workspace = repo.parent
    # Explicitly read applicable physics record requirements before launch.
    for f in (
        workspace / "AGENTS.md",
        workspace / "cap-sim/docs/experiment-records.md",
    ):
        if not f.is_file():
            raise RuntimeError("Missing experiment requirements: " + str(f))
        f.read_text()
    import mujoco
    from humanoid_mimicgen.augmentation.four_task_navigation import prepare_plan
    from humanoid_mimicgen.augmentation.direct_grasp import initialize
    from humanoid_mimicgen.workflow.contracts import file_hash

    stamp = now()
    day = stamp[:10]
    ident = (
        day.replace("-", "")
        + "_"
        + stamp[11:16].replace(":", "")
        + "_experience_"
        + task
        + ("_负10" if offset < 0 else "_正10")
    )
    parent = workspace / "experiments" / day[:7] / day[8:10]
    parent.mkdir(parents=True, exist_ok=True)
    case = parent / ident
    sequence = 1
    while case.exists():
        case = parent / (ident + "_" + str(sequence))
        sequence += 1
    case.mkdir()
    (case / "execution").mkdir()
    plan_path = prepare_plan(case, task, offset)
    plan = json.loads(plan_path.read_text())
    prior = extract(runtime, task, plan["source_manifest"])
    save(case / "prior.json", prior)
    save(case / "extraction_evidence.json", prior["extraction_evidence"])
    for file, key in [
        ("scene.mjb", "model"),
        ("initial.npz", "initial"),
        ("task.json", "task"),
        ("physics.json", "physics"),
    ]:
        shutil.copy2(plan["scene_files"][key], case / file)
    shutil.copy2(plan_path, case / "offline_plan.json")
    cfg = dict(
        strategy=plan["execution_task"]["strategy"],
        navigation={"enabled": True},
        task_family=task,
        repository=str(runtime),
    )
    save(case / "config.json", cfg)
    a, _ = initialize(case)
    checker = Path(__import__("inspect").getfile(type(a.b.e.task)))
    checkpoint_probe(a, prior, case)
    runtime_path = case / "runtime.json"
    subprocess.run(
        [
            str(workspace / "cap-sim/scripts/humano_python.sh"),
            str(runtime / "scripts/check_humano_runtime.py"),
            "--out",
            str(runtime_path),
        ],
        check=True,
        stdout=(case / "runtime.log").open("w"),
    )
    rt = json.loads(runtime_path.read_text())
    rt.update(
        actual_session_providers=a.providers,
        sessions_loaded=True,
        actual_loaded_model_files=a.model_files,
    )
    save(runtime_path, rt)
    if (
        rt["versions"]["onnxruntime"] != "1.23.2"
        or rt["versions"]["tensorrt"] != "10.13.0.35"
        or any(p != ["CPUExecutionProvider"] for p in a.providers.values())
    ):
        raise RuntimeError("Pinned runtime violation")
    np.savez_compressed(case / "controller_initial.npz", **controller_arrays(a.sonic))
    save(case / "initial_observation.json", a.observe())
    comparison = dict(
        baseline=prior["provenance"]["reference_baseline"],
        source_manifest=plan["source_manifest"],
        scene=plan["execution_task"]["scene_comparison"],
        declared_differences=[
            "Observation-conditioned spatial phase generation from three compressed exemplar poses",
            "Contact gates closure; native task success and retained proof lift gate completion",
            "Live SONIC planner for legs throughout; no body source reference",
            "At most2 exact preapproach checkpoint branches: side approach or deeper contact from failure evidence",
            "Source initial history and unchanged scene physics, target object lateral translation only",
            "Navigation posture rigidly follows measured root; spatial slew applies only to task poses",
            "Recovery uses measured no-acquisition/slip to override lift-timeout labels; verified promotion requires valid physical audit",
        ],
        matched_controller_comparison=False,
    )
    save(case / "comparison_diff.json", comparison)
    snapshot = case / "source_snapshot"
    snapshot.mkdir()
    hashes = {}
    files = dependency_sources(repo, runtime, a.b.e.task)
    for i, f in enumerate(files):
        dest = snapshot / (str(i) + "_" + f.name)
        shutil.copy2(f, dest)
        hashes[str(f)] = sha(f)
        hashes[str(dest)] = sha(dest)
    hashes.update(a.model_files)
    for f in (
        case / "prior.json",
        case / "extraction_evidence.json",
        case / "config.json",
        case / "initial.npz",
        case / "scene.mjb",
        case / "task.json",
        case / "physics.json",
        case / "controller_initial.npz",
        runtime_path,
        case / "initial_observation.json",
        case / "comparison_diff.json",
        case / "checkpoint_preflight.json",
    ):
        hashes[str(f)] = sha(f)
    hashes.update(plan["evidence_sha256"])
    c = json.loads(
        (workspace / "cap-sim/docs/experiment-record.template.json").read_text()
    )
    c.update(record_status="frozen_ready", experiment_id=case.name, created_at=now())
    c["purpose"] = dict(
        hypothesis="Compressed spatial experience with measured phase gates can complete full navigation and task at new lateral offset",
        baseline_paths=[plan["source_manifest"]],
        planned_changes=comparison["declared_differences"],
        invariants=[
            "unchanged physics and native checker",
            "SONIC1.0 CPU mode1; separate synchronized native Dex3 hands",
            "no original tokens, mode0, decoder override or teleport",
        ],
        known_confounders=[
            "New strategy, not matched controller replacement",
            "Three demonstration keyframes visible to extraction; executor cannot read action sequence",
            "Source baseline available but no new paired replay baseline launched",
        ],
    )
    c["cases"] = [
        dict(
            task=task,
            offset_m=offset,
            initialization="original far robot state and real source SONIC history",
        )
    ]
    c["runtime"].update(
        actual_loaded_session_evidence=str(runtime_path),
        model_hashes=a.model_files,
        python_version=sys.version,
        gear_sonic_package_version=rt["versions"]["gear_sonic"],
        render_device="disabled; numeric validation",
    )
    c["physics"].update(
        simulator_version=mujoco.__version__,
        robot_and_hand="source G1 bilateral Dex3",
        asset_and_initial_state_manifest={
            str(case / f): sha(case / f)
            for f in ("scene.mjb", "initial.npz", "controller_initial.npz")
        },
        physics_parameters_manifest=str(case / "physics.json"),
        physics_dt_s=a.dt,
        control_dt_s=0.02,
        pd_parameters=dict(
            kp=a.sonic.kp.tolist(), kd=a.sonic.kd.tolist(), hands="native hand_command"
        ),
        action_permissions="world wrist/torso targets; planner velocity/height/facing;14 hand angles; complete checkpoint restore only",
        safety_gates={"nonfinite": "stop; no arbitrary collision veto"},
        acceptance_source_and_hash={
            str(path): sha(path)
            for path in {
                checker,
                *[
                    Path(__import__("inspect").getfile(getattr(a.b.e.task, name)))
                    for name in ("check_success", "compute_reward")
                ],
            }
        },
        acceptance_thresholds=dict(
            native_task_predicate="unchanged",
            retention_applicable=task in ("apple", "plate"),
            retention=dict(clearance_m=0.03, hand_N=0.1, support_N=0.1),
            hold_s=1.0,
            button_approval="independent historical20N/1mm/50tick profile and envelope; native reported separately",
        ),
    )
    c["control"].update(
        stage_interfaces=[
            dict(
                stage=s,
                mode="SONIC encoder mode1",
                lower_body_interface="live published planner; no reference replay",
                upper_body_interface="object-relative phase targets with measured spatial slew",
                action_fields_units_frames="world xyz m/wxyz;velocity m/s;absolute yaw rad;height m;fingers14rad",
                target_source=str(case / "prior.json"),
                interpolation=(
                    "Measured-root rigid attachment during navigation;50Hz references held200Hz"
                    if s == "navigation"
                    else "bounded translation0.2m/s rotation0.8rad/s;50Hz references held200Hz"
                ),
                ik_used=False,
                native_prefix_source_and_hash={"duration_s": 0, "source": "none"},
                handoff_rule="measured distance/support; pose/contact; native proof; see frozen program.py",
            )
            for s in (
                "navigation",
                "settle",
                "approach",
                "contact",
                "acquire",
                "lift",
                "hold",
            )
        ],
        actual_mode_log=str(case / "execution/controller.json"),
        original_token_steps=0,
        action_recording={
            "body": "all actual encoder tokens",
            "hands": "14 values every physics step",
            "mapping": "execution/sonic_actions.npz",
        },
    )
    c["pipeline"].update(
        entrypoint="scripts/run.sh batch or run " + str(case),
        ordered_steps=[
            "extract offline spatial keyframes",
            "prepare scene and load pinned sessions",
            "freeze complete evidence",
            "execute own numeric program",
            "classify feedback and restore complete checkpoint up to2",
            "save all attempts and body tokens/hands",
        ],
        source_snapshot_and_hashes=hashes,
        git_commit_and_dirty_patch={
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "source_hashes": hashes,
        },
        skill_versions_and_hashes=[],
        planner_identity_and_prompt_hash={
            "identity": "session-authored numeric phase program",
            "source_sha256": sha(repo / "src/experience_agent/program.py"),
        },
        external_model_calls=0,
        original_actions_visible={"extraction": True, "execution": False},
        feedback_and_reflection_rules="Measured object-frame error directs branch1(1.5cm); orthogonal branch2(2.5cm); task-specific grasp/seat/press alignment; nonfinite unsupported; max2",
        allowed_retry_changes=[
            "deeper_contact",
            "side_approach",
            "higher_proof_lift",
            "seat_alignment",
            "press_alignment",
        ],
        optimization_enabled=False,
    )
    c["budget"] = dict(
        attempts_per_case=3,
        recovery_attempts=2,
        simulation_seconds_per_attempt=80.0,
        feedback_calls="50Hz deterministic",
        optimization_attempts=0,
    )
    c["metrics"] = dict(
        target_definition="current measured object+compressed phase relation; lift uses acquisition anchor",
        sampling_and_alignment="precontrol and poststate every5ms; token index+14hands every step",
        tracking_error_definition="poststate minus applied world target; raw and slew target retained",
        endpoint_error_definition="last executed target; unexecuted stages excluded",
        orientation_error_definition="relative SO3 angle rad",
        base_drift_definition="root world displacement; full root state retained",
        failure_truncation_rule="failed suffixes separate branches; exact prefix restored",
    )
    c["preflight"] = dict(
        passed=True,
        comparison_diff_path=str(case / "comparison_diff.json"),
        evidence_paths=list(hashes),
        evidence_sha256=hashes,
    )
    c["execution"] = dict(
        started_at="execution_audit.json",
        ended_at="execution_audit.json",
        contract_sha256="experiment.sha256",
        audit_path="execution_audit.json",
        timing_path="execution_audit.json",
        results_path="result.json",
    )
    save(case / "experiment.json", c)
    (case / "experiment.sha256").write_text(sha(case / "experiment.json"))
    return case, a, prior


def dependency_sources(repo, runtime, task):
    import inspect

    repo = Path(repo).resolve()
    runtime = Path(runtime).resolve()
    workspace = repo.parent
    files = (
        set((repo / "src").rglob("*.py"))
        | set((runtime / "humanoid_mimicgen").rglob("*.py"))
        | set((runtime / "src/cap_sim").rglob("*.py"))
    )
    files.update(
        [
            repo / "pyproject.toml",
            workspace / "AGENTS.md",
            workspace / "cap-sim/docs/experiment-records.md",
            workspace / "cap-sim/docs/experiment-record.template.json",
            workspace / "cap-sim/scripts/humano_python.sh",
            workspace / "cap-sim/scripts/check_humano_runtime.py",
            runtime / "scripts/humano_python.sh",
            runtime / "scripts/check_humano_runtime.py",
            runtime / "runtime/humano-cpu.json",
        ]
    )
    if task is not None:
        for cls in type(task).__mro__:
            try:
                files.add(Path(inspect.getfile(cls)))
            except (TypeError, OSError):
                pass
        for name in ("check_success", "compute_reward"):
            if hasattr(task, name):
                files.add(Path(inspect.getfile(getattr(task, name))))
        for module in tuple(sys.modules.values()):
            path = getattr(module, "__file__", None)
            if (
                path
                and "/HUMANO/src/humano/" in str(path)
                and str(path).endswith(".py")
            ):
                files.add(Path(path))
    return sorted(f for f in files if f.is_file())


def verify_sources(evidence):
    for path, expected in evidence.items():
        if sha(path) != expected:
            raise ValueError("Frozen evidence changed: " + str(path))


def checkpoint_probe(a, prior, case):
    # No physics: exercise real resources, serialization and identical next
    # policy inference before launch, then restore original complete state.
    import gzip, pickle
    from .backends.sonic import Checkpoint
    from .program import Program

    cp = Checkpoint(a, Program(prior))
    blob = pickle.dumps(cp)
    restored = pickle.loads(blob)
    poses = a.poses()
    f = np.asarray(prior["open"])
    height = float(a.d.qpos[2])
    a.command(poses, np.zeros(2), 0.0, height, f, policy_tick=True)
    target = a.last_target.copy()
    torque = a.d.ctrl.copy()
    token = np.asarray(a.sonic.records[-1]["token"])
    restored.restore(a)
    a.command(poses, np.zeros(2), 0.0, height, f, policy_tick=True)
    if (
        not np.array_equal(torque, a.d.ctrl)
        or not np.array_equal(target, a.last_target)
        or not np.array_equal(token, a.sonic.records[-1]["token"])
    ):
        raise RuntimeError("Real checkpoint next-command continuity failed")
    restored.restore(a)
    save(
        case / "checkpoint_preflight.json",
        dict(
            physics_steps=0,
            real_native_task=True,
            serialized_bytes=len(blob),
            integration_exact=True,
            next_command_exact=True,
            next_token_exact=True,
            history_size=len(a.sonic.history),
        ),
    )
