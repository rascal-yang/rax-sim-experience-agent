import numpy as np
from scipy.spatial.transform import Rotation


class Recovery:
    def __init__(self):
        self.used = []

    def propose(self, reason, observation):
        reasons = {
            "slip",
            "no_acquisition",
            "contact_timeout",
            "acquire_timeout",
            "approach_timeout",
            "insufficient_lift",
            "lift_timeout",
            "task_predicate_failed",
            "task_predicate_lost",
            "navigation_timeout",
        }
        if len(self.used) >= 2 or reason not in reasons:
            return None
        o = observation
        task = o.get("task", "apple")
        q = np.asarray(o["object_pose"])
        r = Rotation.from_quat(np.roll(q[3:], -1))
        sides = ["left", "right"] if task == "plate" else ["right"]
        errors = o.get("hand_target_error_world_m", {})
        contacts = o.get("hand_contact_N", {})
        failed = min(
            sides,
            key=lambda side: (
                contacts.get(side, 0.0),
                -np.linalg.norm(errors.get(side, [0.0, 0.0, 0.0])),
            ),
        )
        measured = np.asarray(o.get(failed + "_pose", o["root_pose"]))[:3]
        error = r.inv().apply(errors.get(failed, q[:3] - measured))
        error[2] = 0.0
        n = np.linalg.norm(error)
        direction = error / max(n, 1e-9) if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        index = len(self.used)
        change = (
            "seat_alignment"
            if task == "chair"
            else (
                "press_alignment"
                if task == "button"
                else (
                    "deeper_contact"
                    if reason in ("slip", "insufficient_lift", "acquire_timeout")
                    else "side_approach"
                )
            )
        )
        # Branch1 follows measured horizontal error; branch2 tries orthogonal
        # approach, preserving evidence and explicit magnitude rather than gain.
        offset = (
            0.015 * direction
            if index == 0
            else 0.025 * np.array([-direction[1], direction[0], 0.0])
        )
        if reason in ("insufficient_lift", "lift_timeout"):
            change = "higher_proof_lift"
            offset = np.array([0.0, 0.0, 0.03 if index == 0 else 0.05])
        elif change == "deeper_contact":
            offset = (0.015 if index == 0 else 0.025) * direction + np.array(
                [0.0, 0.0, -0.008 if index == 0 else -0.012]
            )
        elif change == "press_alignment":
            offset[2] = -0.005 if index == 0 else -0.008
        elif change == "seat_alignment":
            offset[2] = -0.015 if index == 0 else -0.025
        proposal = dict(
            change=change,
            active_sides=[failed],
            offset_object_m=offset.tolist(),
            failure=reason,
            evidence=o,
            branch=index + 1,
        )
        self.used.append(proposal)
        return proposal


def recovery_reason(stage_failure, observed_outcome):
    if stage_failure in ("lift_timeout", "acquire_timeout") and observed_outcome in (
        "no_acquisition",
        "slip",
    ):
        return observed_outcome
    return stage_failure
