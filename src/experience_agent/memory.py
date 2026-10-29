from pathlib import Path
import json
from .contracts import save, now


class Memory:
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        return (
            json.loads(self.path.read_text())
            if self.path.exists()
            else dict(version=0, entries=[])
        )

    def record(self, task, outcome, *, verified, provenance=None):
        x = self.read()
        x["version"] += 1
        x["entries"].append(
            dict(
                task=task,
                outcome=outcome,
                verified_success=bool(
                    verified
                    and outcome.get("success") is True
                    and outcome.get("record_valid") is True
                    and outcome.get("audit", {}).get("passed") is True
                    and outcome.get("physical_evidence_kind") == "native_mujoco"
                    and outcome.get("actual_integrated_steps_all_branches", 0) > 0
                ),
                recorded_at=now(),
                provenance=provenance,
                applicability="task and declared lateral condition only",
            )
        )
        save(self.path, x)
