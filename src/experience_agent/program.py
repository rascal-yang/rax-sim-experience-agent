"""Finite numeric program. No source row, replay clock or action sequence access.

Only phase relations and hand exemplars cross the extraction boundary. Current
object measurements regenerate every approach target, including after contact.
"""

import copy
import numpy as np
from scipy.spatial.transform import Rotation


def classify(task, o):
    if task in ("chair", "button"):
        return (
            "success"
            if (
                o.get("approved_success", o["native_success"])
                if task == "button"
                else o["native_success"]
            )
            else "task_predicate_failed"
        )
    c = o["contacts"]
    if o["clearance_m"] > 0.03 and c["hand_object_N"] <= 0.1:
        return "slip"
    if o["native_success"] and c["hand_object_N"] > 0.1 and c["support_N"] < 0.1:
        return "success"
    return "no_acquisition" if c["hand_object_N"] <= 0.1 else "insufficient_lift"


class Program:
    def __init__(self, prior, branch=None):
        self.p = copy.deepcopy(prior)
        self.stage = "navigation"
        self.elapsed = 0.0
        self.quiet = 0.0
        self.contact = 0.0
        self.held = 0.0
        self.failure = None
        self.button_acceptance = None
        if prior["task"] == "button" and "button_limits" in prior:
            from .acceptance import ButtonAcceptance

            self.button_acceptance = ButtonAcceptance(prior["button_limits"])
        self.branch = branch or {}
        self.lift_anchor = None
        self.transitions = []
        self.standing_height = None

    def transition(self, stage, o):
        self.transitions.append(
            dict(from_stage=self.stage, to_stage=stage, time_s=o["time_s"])
        )
        self.stage = stage
        self.elapsed = 0.0
        self.quiet = 0.0

    def step(self, o, dt):
        self.elapsed += dt
        p = self.p
        task = p["task"]
        if self.button_acceptance is not None and "button_contact" in o:
            o["approved_success"] = self.button_acceptance.update(o["button_contact"])
        if self.standing_height is None:
            self.standing_height = float(o["root_pose"][2])
        obj = np.asarray(o["object_pose"][:3])
        goal = obj[:2] + np.asarray(p["stance"][:2])
        delta = goal - np.asarray(o["root_pose"][:2])
        dist = np.linalg.norm(delta)
        phase = (
            "approach"
            if self.stage in ("navigation", "settle", "approach")
            else "contact" if self.stage in ("contact", "acquire") else "lift"
        )
        anchor = (
            obj
            if self.lift_anchor is None or self.stage not in ("lift", "hold", "done")
            else self.lift_anchor
        )
        poses = {k: np.asarray(v, float).copy() for k, v in p["poses"][phase].items()}
        for k in poses:
            poses[k][:3] += anchor
        # Bounded evidence vector in the current object frame, rotated to world.
        correction = Rotation.from_quat(
            np.roll(np.asarray(o["object_pose"][3:]), -1)
        ).apply(self.branch.get("offset_object_m", [0.0, 0.0, 0.0]))
        apply = self.branch.get("change") != "higher_proof_lift" or self.stage in (
            "lift",
            "hold",
            "done",
        )
        for side in self.branch.get("active_sides", p["active"]):
            if apply:
                poses[side][:3] += correction
        if task == "chair":
            for side in poses:
                poses[side][:3] += correction
        fingers = (
            p["closed"]
            if self.stage in ("acquire", "lift", "hold", "done")
            else p["open"]
        )
        if task == "button" and self.stage in ("contact", "hold", "done"):
            fingers = p.get("contact_fingers", p["closed"])
        stable = (
            min(o["foot_ground_N"]) > 10
            and o["root_speed_m_s"] < 0.08
            and o["yaw_speed_rad_s"] < 0.3
        )
        near = all(
            np.linalg.norm(poses[s][:3] - np.asarray(o[s + "_pose"][:3])) < 0.035
            for s in p["active"]
        )
        velocity = np.zeros(2)
        if self.stage == "navigation":
            velocity = (
                delta
                / max(dist, 1e-9)
                * (min(0.5, max(0.25, dist * 1.5)) if dist >= 0.1 else 0.0)
            )
            if dist < 0.1 and o["root_speed_m_s"] < 0.15:
                self.transition("settle", o)
        elif self.stage == "settle":
            self.quiet = self.quiet + dt if stable else 0.0
            if self.quiet >= 0.3:
                self.transition("approach", o)
        elif self.stage == "approach":
            if (near and stable) or (task == "chair" and stable):
                self.transition("contact", o)
        elif self.stage == "contact":
            contact = (
                o["contacts"]["hand_object_N"] > 0.1
                if task in ("apple", "plate")
                else o["contacts"]["robot_object_N"] > 0.1
            )
            self.contact = self.contact + dt if contact else 0.0
            if task in ("apple", "plate") and self.contact >= 0.08:
                fingers = p["closed"]
                self.transition("acquire", o)
            elif task in ("chair", "button") and classify(task, o) == "success":
                self.transition("hold", o)
        elif self.stage == "acquire":
            if o["contacts"]["hand_object_N"] > 0.1 and stable:
                self.quiet += dt
                if self.quiet >= 0.3:
                    self.lift_anchor = obj.copy()
                    self.transition("lift", o)
            else:
                self.quiet = 0.0
        elif self.stage == "lift":
            if classify(task, o) == "slip":
                self.failure = "slip"
            elif classify(task, o) == "success":
                self.transition("hold", o)
        elif self.stage == "hold":
            if classify(task, o) != "success":
                self.failure = (
                    "slip" if task in ("apple", "plate") else "task_predicate_lost"
                )
            else:
                self.held += dt
                if self.held >= 1.0:
                    self.transition("done", o)
        if (
            self.elapsed
            > dict(
                navigation=45,
                settle=8,
                approach=8,
                contact=8,
                acquire=4,
                lift=8,
                hold=3,
                done=1,
            )[self.stage]
        ):
            self.failure = self.stage + "_timeout"
        height = (
            self.standing_height
            if self.stage in ("navigation", "settle", "approach")
            else p["height"]
        )
        if task == "chair" and self.stage in ("contact", "hold", "done"):
            height = p.get("seated_height", 0.5) + float(correction[2])
        return dict(
            stage=self.stage,
            poses={s: v.tolist() for s, v in poses.items()},
            velocity=velocity.tolist(),
            facing=p["stance"][2],
            height=height,
            fingers=list(fingers),
            failure=self.failure,
            completed=self.stage == "done",
            distance_m=float(dist),
        )
