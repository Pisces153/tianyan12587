from __future__ import annotations

from scripts import run_b4_map_root_cause_ab as module


def test_legacy_control_is_five_endpoint_all_pass() -> None:
    rows = module.legacy_artifact_rows()
    assert {row["endpoint"] for row in rows} == set(module.regrid.ENDPOINT_NAMES)
    assert all(row["joint_pass"] for row in rows)
    worth = next(row for row in rows if row["endpoint"] == "worth_sensing_map")
    assert worth["power"] == 0.838


def test_classify_splits_regression_from_new_regime() -> None:
    legacy = [
        {
            "size_exact_match": True,
            "power_exact_match": True,
            "replay_joint_pass": True,
        }
    ]
    corrected = [
        {"endpoint": "worth_sensing_map", "joint_pass": True}
        for _ in module.SESSION_GAPS
    ]
    assert module.classify(legacy, corrected)["verdict"] == "pipeline_intact_new_R_c_regime_effect_supported"
    corrected[0]["joint_pass"] = False
    assert "requires_bisection" in module.classify(legacy, corrected)["verdict"]


def test_seed_is_frozen(tmp_path) -> None:
    try:
        module.run(
            output=tmp_path / "bad",
            replicates=1,
            refinement_replicates=1,
            seed=1,
            workers=1,
        )
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("non-frozen seed was accepted")
