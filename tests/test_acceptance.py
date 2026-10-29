from experience_agent.acceptance import ButtonAcceptance


def test_button_requires_exact_50tick_force_distance_hold_and_envelope():
    a = ButtonAcceptance(
        {
            "root_height_min_m": 0.5,
            "root_tilt_max_rad": 0.5,
            "robot_table_force_N": 50.0,
            "selected_contact_force_N": 200.0,
        }
    )
    row = dict(
        force_N=21.0,
        distance_m=0.0005,
        contacts=1,
        root_height_m=0.75,
        root_tilt_rad=0.0,
        table_force_N=0.0,
        selected_force_N=21.0,
        warning_count=0,
    )
    for _ in range(49):
        assert not a.update(row)
    assert a.update(row)
    a.update(row | {"table_force_N": 60.0})
    assert not a.success
    b = ButtonAcceptance(a.limits)
    for _ in range(60):
        assert not b.update(row | {"force_N": 19.0})
