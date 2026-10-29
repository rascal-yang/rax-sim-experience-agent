"""Approved button predicate, independent of the native checker.

Thresholds are the reviewed20N/1mm/50policy-tick rule. Envelope is the
historical source's declared safety envelope, never a physics abort gate.
"""

import numpy as np


class ButtonAcceptance:
    def __init__(self, limits):
        self.limits = limits
        self.count = 0
        self.longest = 0
        self.envelope = True
        self.success = False

    def update(self, row):
        good = (
            row["contacts"] >= 1
            and row["distance_m"] <= 0.001
            and row["force_N"] >= 20.0
        )
        self.count = self.count + 1 if good else 0
        self.longest = max(self.longest, self.count)
        o = self.limits
        self.envelope = (
            self.envelope
            and row["root_height_m"] >= o["root_height_min_m"]
            and row["root_tilt_rad"] <= o["root_tilt_max_rad"]
            and row["table_force_N"] <= o["robot_table_force_N"]
            and row["selected_force_N"] <= o["selected_contact_force_N"]
            and row["warning_count"] == 0
        )
        self.success = bool(self.longest >= 50 and self.envelope)
        return self.success


def button_contact(a, ob, selected_names):
    import mujoco
    from humanoid_mimicgen.augmentation.button_acceptance import accepts_contact

    m, d = a.m, a.d
    force = 0.0
    distance = 1e9
    contacts = 0
    selected = 0.0
    for i, c in enumerate(d.contact):
        gs = [int(c.geom1), int(c.geom2)]
        buttons = [
            g
            for g in gs
            if any(
                k in (m.geom(g).name or "") for k in ("E_button_01_13", "E_button_02_6")
            )
        ]
        if not buttons:
            continue
        button = buttons[0]
        other = gs[1] if button == gs[0] else gs[0]
        body = m.body(int(m.geom_bodyid[other])).name or ""
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        if body in selected_names:
            selected += max(0.0, float(f[0]))
        mesh = (
            (m.mesh(int(m.geom_dataid[other])).name or "")
            if m.geom_type[other] == mujoco.mjtGeom.mjGEOM_MESH
            else ""
        )
        if not accepts_contact(body, mesh, True):
            continue
        world = np.asarray(c.frame).reshape(3, 3).T @ f[:3]
        force += max(0.0, float(world[2] if button == gs[0] else -world[2]))
        contacts += 1
        distance = min(distance, float(c.dist))
    q = ob["root_pose"]
    table = sum(
        e["force_N"]
        for e in ob["contacts"]["events"]
        if "table" in e["bodies"]
        and any(b in {m.body(i).name for i in a.robot_bodies} for b in e["bodies"])
    )
    return dict(
        force_N=force,
        distance_m=distance,
        contacts=contacts,
        root_height_m=q[2],
        root_tilt_rad=float(np.arccos(np.clip(1 - 2 * (q[4] ** 2 + q[5] ** 2), -1, 1))),
        table_force_N=table,
        selected_force_N=selected,
        warning_count=int(sum(w.number for w in d.warning)),
    )
