#!/usr/bin/env python3
"""Read saved evidence only; no model loading or physics calls."""

import json
from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    x = np.asarray(values, float)
    return (
        dict(
            rmse=float(np.sqrt(np.mean(x * x))),
            p95=float(np.percentile(x, 95)),
            maximum=float(x.max()),
        )
        if len(x)
        else None
    )


def main():
    batch = json.loads((ROOT / "docs/validation-results.json").read_text())
    rows = []
    for entry in batch["conditions"]:
        p = Path(entry["case"])
        result = json.loads((p / "result.json").read_text())
        observations = json.loads((p / "execution/observations.json").read_text())
        commands = json.loads((p / "execution/commands.json").read_text())
        prior = json.loads((p / "prior.json").read_text())
        source = json.loads(
            Path(
                json.loads((p / "offline_plan.json").read_text())["source_manifest"]
            ).read_text()
        )
        import mujoco

        model = mujoco.MjModel.from_binary_path(str(p / "scene.mjb"))
        root = model.body(prior["bindings"]["root_body"]).id
        robot_names = set()
        for bid in range(model.nbody):
            ancestor = bid
            while ancestor and ancestor != root:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == root:
                robot_names.add(model.body(bid).name)
        all_coverage = {
            s: 0
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
        }
        for attempt in result["attempts"]:
            path = p / "branches" / str(attempt["branch"]) / "commands.json"
            branch_commands = (
                json.loads(path.read_text()) if path.exists() else commands
            )
            for command in branch_commands[attempt.get("inherited_prefix_steps", 0) :]:
                all_coverage[command["stage"]] += 1
        errors = {s: [] for s in ("left", "right")}
        angles = {s: [] for s in errors}
        table = obstacle = 0.0
        for ob, cmd in zip(observations, commands):
            for side in errors:
                measured = np.asarray(ob[side + "_pose"])
                target = np.asarray(cmd["encoder_pose"][side])
                errors[side].append(float(np.linalg.norm(target[:3] - measured[:3])))
                angles[side].append(
                    float(
                        (
                            Rotation.from_quat(np.roll(target[3:], -1)).inv()
                            * Rotation.from_quat(np.roll(measured[3:], -1))
                        ).magnitude()
                    )
                )
            obstacle = max(obstacle, ob["contacts"]["robot_obstacle_N"])
            table = max(
                table,
                sum(
                    e["force_N"]
                    for e in ob["contacts"]["events"]
                    if "table" in e["bodies"]
                    and any(b in robot_names for b in e["bodies"])
                ),
            )
        summary = dict(
            experiment_id=p.name,
            task=entry["task"],
            offset_m=entry["offset_m"],
            native_ever=result["native_success"],
            native_terminal=result["native_terminal"],
            success=result["success"],
            failure=result["failure"],
            completed=result["completed"],
            stage_coverage=result.get("stage_coverage"),
            stage_coverage_all_integrated_suffixes=all_coverage,
            attempts=result["attempts"],
            actual_integrated_steps_all_branches=result.get(
                "actual_integrated_steps_all_branches", result["physics_steps"]
            ),
            recorded_final_branch_steps=result["physics_steps"],
            wrist_tracking_m={s: stats(x) for s, x in errors.items()},
            wrist_tracking_rad={s: stats(x) for s, x in angles.items()},
            true_table_contact_peak_N=table,
            all_obstacle_contact_peak_N=obstacle,
            criterion=source["success_criterion"],
            retained_grasp=result["retained_grasp"],
            timing=result["timing"],
            body_hand_audit=result["audit"],
            contract_sha256=hashlib.sha256(
                (p / "experiment.json").read_bytes()
            ).hexdigest(),
            frozen_contract_hash_matches=hashlib.sha256(
                (p / "experiment.json").read_bytes()
            ).hexdigest()
            == (p / "experiment.sha256").read_text().strip(),
        )
        if (p / "button_approval.json").exists():
            summary["button_approval"] = json.loads(
                (p / "button_approval.json").read_text()
            )
        summary["actual_frozen_snapshot_hashes_match"] = all(
            hashlib.sha256(Path(f).read_bytes()).hexdigest() == h
            for f, h in json.loads((p / "experiment.json").read_text())["preflight"][
                "evidence_sha256"
            ].items()
            if "/source_snapshot/" in f
        )
        summary["posthoc_script_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        record = json.loads((p / "experiment.json").read_text())
        (p / "REPORT.md").write_text(
            "# "
            + p.name
            + "\n\n"
            + "Start: "
            + result["started_at"]
            + "; end: "
            + result["ended_at"]
            + "\n\n"
            + "Purpose: observation-conditioned full navigation and "
            + entry["task"]
            + " at Y"
            + str(entry["offset_m"])
            + " m. Historical baseline: "
            + record["purpose"]["baseline_paths"][0]
            + ". New spatial policy differs from trajectory retargeting; no matched controller attribution.\n\n"
            + "Runtime: TRT10.13.0.35 / ORT1.23.2 / SONIC1.0 CPU; source G1/Dex3. All actual encoder modes: "
            + str(result["audit"].get("actual_modes"))
            + ". Separate synchronized14-angle hands; no original prefix or decoder override.\n\n"
            + "Pipeline commit: "
            + str(record["pipeline"]["git_commit_and_dirty_patch"]["commit"])
            + "; frozen contract SHA256: "
            + summary["contract_sha256"]
            + ". Complete runtime/source evidence: experiment.json, source_snapshot/, runtime.json.\n\n"
            + "Result: success="
            + str(result["success"])
            + ", native ever="
            + str(result["native_success"])
            + ", native terminal="
            + str(result["native_terminal"])
            + ", failure="
            + str(result["failure"])
            + ". One condition, "
            + str(len(result["attempts"]))
            + " actual attempts.\n\n"
            + "Stage coverage (physical steps): "
            + json.dumps(result.get("stage_coverage"), ensure_ascii=False)
            + ". Unexecuted task stages do not count as validation.\n\n"
            + "Executed audit: "
            + json.dumps(result["audit"])
            + ".\n\n"
            + "Timing: "
            + json.dumps(result["timing"])
            + ". physics_step_compute excludes SONIC inference; total_execution includes control, inference, logging and physics. Preparation and rendering unmeasured/disabled.\n\n"
            + "Full numeric measurements and failed branch counts: numeric_diagnostics.json; raw evidence: execution/ and branches/. No verified reusable success is claimed for failed conditions.\n"
        )
        (p / "numeric_diagnostics.json").write_text(json.dumps(summary, indent=2))
        rows.append(summary)
    (ROOT / "docs/validation-diagnostics.json").write_text(
        json.dumps(
            dict(
                started_at=batch["started_at"],
                ended_at=batch["ended_at"],
                conditions=rows,
            ),
            indent=2,
        )
    )
    print(
        json.dumps(
            [
                dict(
                    task=x["task"],
                    offset_m=x["offset_m"],
                    success=x["success"],
                    failure=x["failure"],
                    coverage=x["stage_coverage"],
                )
                for x in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
