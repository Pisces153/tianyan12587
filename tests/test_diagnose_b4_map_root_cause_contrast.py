from scripts import diagnose_b4_map_root_cause_contrast as module


def summary(backend: str, gap: int, power: float, economic: float, floor_right: float) -> dict:
    return {
        "backend_id": backend,
        "session_gap_days": gap,
        "power": power,
        "detection_failure_rate": 0.0,
        "ou_fit_failure_rate": 0.0,
        "economic_separation_failure_rate": economic,
        "floor_right_of_t_star_rate": floor_right,
    }


def test_classify_expected_mechanism() -> None:
    rows = []
    for gap in module.regrid.SESSION_GAP_DAY_NUISANCE:
        rows.append(summary("old_peak", gap, 0.84, 0.16, 0.1))
        rows.append(summary("old_boundary", gap, 0.80, 0.20, 0.4))
        rows.append(summary("tianyan-287_measured", gap, 0.70, 0.30, 0.9))
        rows.append(summary("tianyan176_measured", gap, 0.75, 0.25, 0.8))
    assert module.classify(rows)["verdict"].startswith("high_fixed_overhead")


def test_analytic_floor_increases_with_overhead() -> None:
    analytic = [module.analytic_profile(profile) for profile in module.profiles()]
    by_name = {row["backend_id"]: row for row in analytic}
    assert by_name["old_peak"]["design_floor_seconds"] < by_name["old_boundary"]["design_floor_seconds"]
    assert by_name["old_boundary"]["design_floor_seconds"] < by_name["tianyan-287_measured"]["design_floor_seconds"]
