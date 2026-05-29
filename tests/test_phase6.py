import copy
import json
import os

import pytest

from phase6_quotation import (
    MATERIAL_PROPERTIES,
    check_axis_capability,
    check_capacity,
    check_material_available,
    check_tool_reach,
    check_tooling_available,
    check_work_envelope,
    compute_cost,
    generate_quotation,
    select_machine,
)


CLI_DIR = "data/processed/simple_block_cli"
NASH_PROFILE = "factory_profiles/nash_nz.json"
BASIC_PROFILE = "factory_profiles/generic_3axis.json"

MINIMAL_FACTORY = {
    "factory_name": "Test Shop",
    "currency": "NZD",
    "machines": [
        {
            "id": "VMC_01",
            "type": "VMC",
            "axes": 3,
            "work_envelope_mm": {"x": 500, "y": 400, "z": 300},
            "achievable_Ra_um": 1.6,
            "hourly_rate": 120.0,
        }
    ],
    "materials_available": ["aluminium_6061", "mild_steel"],
    "weekly_capacity_hours": 40,
    "overhead_factor": 1.10,
}

MINIMAL_TIME = {
    "total_time_min": 45.0,
    "machining_time_min": 30.0,
    "setup_time_min": 15.0,
    "setup_count": 1,
}

MINIMAL_META = {
    "bounding_box_mm": {"x": 100.0, "y": 60.0, "z": 40.0},
    "volume_mm3": 180000.0,
    "raw_stock_mm": {"x": 115.0, "y": 70.0, "z": 45.0},
}

MINIMAL_PLAN = {
    "axis_requirement": 3,
    "setup_count": 1,
    "operation_count": 4,
}


@pytest.fixture
def input_files(tmp_path):
    factory_path = tmp_path / "factory.json"
    time_path = tmp_path / "time_estimate.json"
    meta_path = tmp_path / "metadata.json"
    plan_path = tmp_path / "process_plan.json"
    factory_path.write_text(json.dumps(MINIMAL_FACTORY))
    time_path.write_text(json.dumps(MINIMAL_TIME))
    meta_path.write_text(json.dumps(MINIMAL_META))
    plan_path.write_text(json.dumps(MINIMAL_PLAN))
    return {
        "factory": str(factory_path),
        "time": str(time_path),
        "meta": str(meta_path),
        "plan": str(plan_path),
        "out": str(tmp_path),
    }


def test_material_properties_all_have_density():
    for _, props in MATERIAL_PROPERTIES.items():
        assert "density_g_per_mm3" in props
        assert props["density_g_per_mm3"] > 0


def test_material_properties_all_have_price():
    for _, props in MATERIAL_PROPERTIES.items():
        assert "price_per_kg" in props
        assert props["price_per_kg"] > 0


def test_axis_check_pass():
    result = check_axis_capability(3, MINIMAL_FACTORY["machines"])
    assert result["pass"] is True
    assert result["required"] == 3


def test_axis_check_fail():
    result = check_axis_capability(5, MINIMAL_FACTORY["machines"])
    assert result["pass"] is False
    assert "reason" in result


def test_envelope_check_pass():
    result = check_work_envelope(
        {"x": 100.0, "y": 60.0, "z": 40.0}, MINIMAL_FACTORY["machines"], required_axes=3
    )
    assert result["pass"] is True


def test_envelope_check_fail_oversized():
    result = check_work_envelope(
        {"x": 600.0, "y": 500.0, "z": 400.0},
        MINIMAL_FACTORY["machines"],
        required_axes=3,
    )
    assert result["pass"] is False


def test_material_check_pass():
    result = check_material_available("aluminium_6061", MINIMAL_FACTORY)
    assert result["pass"] is True


def test_material_check_fail():
    result = check_material_available("titanium_grade5", MINIMAL_FACTORY)
    assert result["pass"] is False
    assert "reason" in result


def test_capacity_check_pass():
    result = check_capacity(45.0, MINIMAL_FACTORY)
    assert result["pass"] is True
    assert result["utilisation_fraction"] < 1.0


def test_capacity_check_fail():
    result = check_capacity(3000.0, MINIMAL_FACTORY)
    assert result["pass"] is False


def test_select_machine_returns_cheapest_valid():
    factory = copy.deepcopy(MINIMAL_FACTORY)
    factory["machines"].append(
        {
            "id": "VMC_PREMIUM",
            "type": "VMC",
            "axes": 3,
            "work_envelope_mm": {"x": 600, "y": 500, "z": 400},
            "achievable_Ra_um": 0.4,
            "hourly_rate": 200.0,
        }
    )
    machine = select_machine(factory, required_axes=3, bounding_box_mm={"x": 100, "y": 60, "z": 40})
    assert machine is not None
    assert machine["hourly_rate"] == 120.0


def test_select_machine_returns_none_when_no_fit():
    machine = select_machine(MINIMAL_FACTORY, required_axes=5, bounding_box_mm={"x": 100, "y": 60, "z": 40})
    assert machine is None


def test_compute_cost_total_positive():
    result = compute_cost(
        MINIMAL_TIME, MINIMAL_META, MINIMAL_FACTORY, "aluminium_6061", MINIMAL_FACTORY["machines"][0]
    )
    assert result["total"] > 0


def test_compute_cost_total_equals_subtotal_times_overhead():
    result = compute_cost(
        MINIMAL_TIME, MINIMAL_META, MINIMAL_FACTORY, "aluminium_6061", MINIMAL_FACTORY["machines"][0]
    )
    expected = result["subtotal"] * MINIMAL_FACTORY["overhead_factor"]
    assert abs(result["total"] - expected) < 0.01


def test_compute_cost_currency_matches_factory():
    result = compute_cost(
        MINIMAL_TIME, MINIMAL_META, MINIMAL_FACTORY, "aluminium_6061", MINIMAL_FACTORY["machines"][0]
    )
    assert result["currency"] == MINIMAL_FACTORY["currency"]


def test_quotation_file_created(input_files):
    generate_quotation(input_files["plan"], input_files["time"], input_files["meta"], input_files["factory"], input_files["out"])
    assert os.path.exists(os.path.join(input_files["out"], "quotation.json"))


def test_quotation_schema_keys(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"], input_files["meta"], input_files["factory"], input_files["out"]
    )
    for key in [
        "recommendation",
        "flags",
        "estimated_cost",
        "time_summary",
        "machine_selected",
        "capability_checks",
        "factory_name",
        "material",
        "axis_required",
        "source_files",
        "quotation_file",
        "warnings",
    ]:
        assert key in result


def test_recommendation_accept_on_capable_factory(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"], input_files["meta"], input_files["factory"], input_files["out"]
    )
    assert result["recommendation"] == "ACCEPT"
    assert result["flags"] == []


def test_recommendation_review_on_25d_incompatibility(input_files, tmp_path):
    review_plan = copy.deepcopy(MINIMAL_PLAN)
    review_plan["two_point_five_d_compatible"] = False
    review_plan["unsupported_reasons"] = ["rectangular_pocket instance 0 requires side access."]
    review_plan["review_items"] = [
        {
            "code": "SIDE_ACCESS_REQUIRED",
            "severity": "review",
            "message": "rectangular_pocket instance 0 requires side access.",
            "source": "phase3_setup_analysis",
        }
    ]
    plan_path = tmp_path / "plan_review.json"
    plan_path.write_text(json.dumps(review_plan))
    result = generate_quotation(
        str(plan_path),
        input_files["time"],
        input_files["meta"],
        input_files["factory"],
        input_files["out"],
    )
    assert result["recommendation"] == "REVIEW"
    assert result["review_required"] is True
    assert "SIDE_ACCESS_REQUIRED" in result["review_codes"]
    assert any("2.5D" in flag or "side access" in flag for flag in result["flags"])


def test_recommendation_reject_axis_mismatch(input_files, tmp_path):
    plan_5axis = copy.deepcopy(MINIMAL_PLAN)
    plan_5axis["axis_requirement"] = 5
    plan_path = tmp_path / "plan5.json"
    plan_path.write_text(json.dumps(plan_5axis))
    result = generate_quotation(
        str(plan_path), input_files["time"], input_files["meta"], input_files["factory"], input_files["out"]
    )
    assert result["recommendation"] == "REJECT"
    assert len(result["flags"]) >= 1


def test_recommendation_reject_wrong_material(input_files):
    result = generate_quotation(
        input_files["plan"],
        input_files["time"],
        input_files["meta"],
        input_files["factory"],
        input_files["out"],
        material="titanium_grade5",
    )
    assert result["recommendation"] == "REJECT"


def test_capability_checks_present(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"], input_files["meta"], input_files["factory"], input_files["out"]
    )
    for check in [
        "axis_capability",
        "work_envelope",
        "material_available",
        "capacity",
        "tooling_available",
        "tool_reach",
    ]:
        assert check in result["capability_checks"]


def test_tooling_check_reports_missing_tool():
    result = check_tooling_available({"tool_list": ["wire_edm"]}, MINIMAL_FACTORY)
    assert result["pass"] is False
    assert "wire_edm" in result["missing"]


def test_tool_reach_check_reports_depth_failure():
    factory = copy.deepcopy(MINIMAL_FACTORY)
    factory["tool_library"] = [{"tool_type": "flat_endmill", "max_depth_mm": 10.0}]
    plan = {
        "feature_feasibility": [
            {"type": "rectangular_pocket", "instance_id": 0, "estimated_depth_mm": 25.0}
        ]
    }
    result = check_tool_reach(plan, factory)
    assert result["pass"] is False
    assert result["failures"][0]["available_reach_mm"] == 10.0


def test_quotation_path_absolute(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"], input_files["meta"], input_files["factory"], input_files["out"]
    )
    assert os.path.isabs(result["quotation_file"])


def test_cost_total_positive(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"], input_files["meta"], input_files["factory"], input_files["out"]
    )
    assert result["estimated_cost"]["total"] > 0


def test_written_json_matches_returned(input_files):
    result = generate_quotation(
        input_files["plan"], input_files["time"], input_files["meta"], input_files["factory"], input_files["out"]
    )
    with open(result["quotation_file"], encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["recommendation"] == result["recommendation"]


def test_missing_process_plan_raises(input_files):
    with pytest.raises(FileNotFoundError):
        generate_quotation("no_plan.json", input_files["time"], input_files["meta"], input_files["factory"], input_files["out"])


def test_missing_factory_profile_raises(input_files):
    with pytest.raises(FileNotFoundError):
        generate_quotation(input_files["plan"], input_files["time"], input_files["meta"], "no_factory.json", input_files["out"])


def test_nash_nz_profile_exists():
    assert os.path.exists(NASH_PROFILE)


def test_generic_3axis_profile_exists():
    assert os.path.exists(BASIC_PROFILE)


def test_nash_nz_profile_valid_json():
    with open(NASH_PROFILE, encoding="utf-8") as f:
        data = json.load(f)
    assert "machines" in data
    assert len(data["machines"]) >= 1


@pytest.mark.skipif(
    not all(os.path.exists(os.path.join(CLI_DIR, filename)) for filename in ["process_plan.json", "metadata.json"])
    or not os.path.exists(NASH_PROFILE),
    reason="Full pipeline CLI outputs or factory profile not available",
)
def test_full_pipeline_simple_block(tmp_path):
    from phase5_time_estimate import estimate_time

    estimate_time(
        os.path.join(CLI_DIR, "process_plan.json"),
        os.path.join(CLI_DIR, "metadata.json"),
        str(tmp_path),
    )
    time_path = os.path.join(str(tmp_path), "time_estimate.json")
    result = generate_quotation(
        os.path.join(CLI_DIR, "process_plan.json"),
        time_path,
        os.path.join(CLI_DIR, "metadata.json"),
        NASH_PROFILE,
        str(tmp_path),
        material="aluminium_6061",
    )
    assert result["recommendation"] in ("ACCEPT", "REVIEW", "REJECT")
    assert result["estimated_cost"]["total"] > 0
