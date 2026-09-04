from scripts import run_b4_map_root_cause_bisection as module


def row(gap: int, passed: bool) -> dict:
    return {"session_gap_days": gap, "joint_pass": passed}


def test_classify_overhead_split() -> None:
    peak = [row(gap, True) for gap in module.SESSION_GAPS]
    boundary = [row(1, False), row(2, True), row(4, True)]
    assert module.classify(peak, boundary)["verdict"].startswith("pipeline_intact_overhead")


def test_classify_gap_split() -> None:
    peak = [row(1, True), row(2, False), row(4, True)]
    boundary = [row(gap, False) for gap in module.SESSION_GAPS]
    assert module.classify(peak, boundary)["verdict"] == "session_gap_semantics_change_power"
