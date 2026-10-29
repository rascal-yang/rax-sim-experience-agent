import numpy as np
import argparse
import json
from pathlib import Path
from .contracts import save, now
from .backends.sonic import dependencies
from .priors import extract
from .runner import run
from .experiment import prepare
from .memory import Memory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["inspect", "batch", "run", "audit"])
    p.add_argument(
        "--task", choices=["apple", "plate", "button", "chair"], default="apple"
    )
    p.add_argument("--offset", type=float, choices=[-0.1, 0.1], default=-0.1)
    p.add_argument(
        "--runtime",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "cap-sim-api-augmentation",
    )
    p.add_argument("--path", type=Path)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.command == "inspect":
        print(json.dumps(extract(args.runtime, args.task), indent=2))
        return
    if args.command == "audit":
        from .runner import audit_actions

        print(
            json.dumps(
                audit_actions(
                    json.loads((args.path / "execution/controller.json").read_text()),
                    json.loads((args.path / "execution/commands.json").read_text()),
                    np.load(args.path / "execution/trajectory.npz"),
                ),
                indent=2,
            )
        )
        return
    jobs = (
        [
            (task, offset)
            for task in ("chair", "apple", "button", "plate")
            for offset in (-0.1, 0.1)
        ]
        if args.command == "batch"
        else [(args.task, args.offset)]
    )
    rows = []
    started = now()
    for task, offset in jobs:
        case, a, prior = prepare(repo, args.runtime, task, offset)
        print("FROZEN " + str(case), flush=True)
        try:
            result = run(case, a, prior)
            # Approved button palm profile is supplementary; preserve native result.
            if task == "button":
                try:
                    from humanoid_mimicgen.augmentation.four_task_navigation import (
                        button_approval,
                    )

                    # Historical tool expects these exact saved-state filenames.
                    result["max_obstacle_N"] = max(
                        (
                            o["contacts"]["robot_obstacle_N"]
                            for o in json.loads(
                                (case / "execution/observations.json").read_text()
                            )
                        ),
                        default=0.0,
                    )
                    save(case / "result.json", result)
                    approval = button_approval(case)
                    save(case / "button_approval.json", approval)
                    result["approved_success"] = approval["approved_success"]
                    result["success"] = (
                        result["success"] and approval["approved_success"]
                    )
                    save(case / "result.json", result)
                except Exception as e:
                    result["approval_error"] = str(e)
                    result["approved_success"] = False
                    result["success"] = False
                    save(case / "result.json", result)
            save(case / "execution_audit.json", result)
            Memory(repo / "experience/memory.json").record(
                task,
                result,
                verified=True,
                provenance={
                    "experiment": str(case),
                    "contract": str(case / "experiment.json"),
                },
            )
            rows.append(dict(case=str(case), task=task, offset_m=offset, result=result))
            print(json.dumps(rows[-1]), flush=True)
        except Exception as e:
            Memory(repo / "experience/memory.json").record(
                task,
                (
                    json.loads((case / "result.json").read_text())
                    if (case / "result.json").exists()
                    else {"failure": repr(e)}
                ),
                verified=False,
                provenance={"experiment": str(case)},
            )
            rows.append(
                dict(
                    case=str(case),
                    task=task,
                    offset_m=offset,
                    error=repr(e),
                    result=(
                        json.loads((case / "result.json").read_text())
                        if (case / "result.json").exists()
                        else None
                    ),
                )
            )
            print(json.dumps(rows[-1]), flush=True)
        save(
            repo / "docs/validation-results.json",
            dict(started_at=started, ended_at=now(), conditions=rows),
        )


if __name__ == "__main__":
    main()
