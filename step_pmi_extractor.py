"""Extract non-tolerancing PMI and approximate feature dimensions from STEP files."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


DEFAULT_RA = {
    "flat_face": 3.2,
    "through_hole": 1.6,
    "blind_hole": 1.6,
    "rectangular_pocket": 3.2,
    "circular_pocket": 3.2,
    "rectangular_slot": 3.2,
    "circular_slot": 3.2,
    "rectangular_step": 3.2,
    "chamfer": 6.3,
    "fillet": 3.2,
    "boss": 3.2,
    "triangular_pocket": 3.2,
}

ROUGH_DOC_MM = {
    "aluminium_6061": 5.0,
    "mild_steel": 2.0,
    "stainless_316": 1.5,
    "titanium_grade5": 0.8,
}

PECK_DEPTH_RATIO = 3.0

MATERIAL_NAME_MAP = {
    "6061": "aluminium_6061",
    "6061-t6": "aluminium_6061",
    "aluminium": "aluminium_6061",
    "aluminum": "aluminium_6061",
    "al": "aluminium_6061",
    "mild steel": "mild_steel",
    "ms": "mild_steel",
    "1018": "mild_steel",
    "316": "stainless_316",
    "316l": "stainless_316",
    "stainless": "stainless_316",
    "ss316": "stainless_316",
    "titanium": "titanium_grade5",
    "ti-6al-4v": "titanium_grade5",
    "grade 5": "titanium_grade5",
}


def _write_json_atomic(data: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, path)


def parse_step_pmi(step_path: str) -> dict:
    """Parse explicit AP242-like PMI annotations from STEP text without raising."""
    result = {
        "material": None,
        "surface_finish": [],
        "threads": [],
    }

    try:
        with open(step_path, errors="ignore") as f:
            content = f.read().upper()
    except Exception:
        return result

    try:
        mat_patterns = [
            r"MATERIAL_DESIGNATION\s*\(\s*'([^']+)'",
            r"MATERIAL\s*\(\s*'([^']+)'",
            r"PRODUCT_DEFINITION_FORMATION\s*\(\s*'([^']+)'",
        ]
        for pattern in mat_patterns:
            match = re.search(pattern, content)
            if match:
                raw = match.group(1).strip().lower()
                for key, normalised in MATERIAL_NAME_MAP.items():
                    if key in raw:
                        result["material"] = normalised
                        break
            if result["material"]:
                break

        ra_pattern = r"SURFACE_TEXTURE_PARAMETER\s*\([^)]*?(\d+\.?\d*)[^)]*\)"
        for match in re.finditer(ra_pattern, content):
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if 0.05 <= value <= 50.0:
                result["surface_finish"].append(
                    {"Ra_um": value, "face_ref": match.group(0)[:40]}
                )

        thread_pattern = (
            r"EXTERNALLY_DEFINED_FEATURE_DEFINITION\s*\(\s*'([^']*(?:M\d|UNC|UNF|NPT|G\d)[^']*)'"
        )
        for match in re.finditer(thread_pattern, content):
            result["threads"].append(
                {"spec": match.group(1).strip(), "face_ref": match.group(0)[:40]}
            )
    except Exception:
        return result

    return result


def measure_brep_features(step_path: str) -> dict:
    """Measure approximate dimensions from B-rep geometry without raising."""
    result = {
        "holes": [],
        "planar_recesses": [],
        "bounding_box_mm": {"x": 0.0, "y": 0.0, "z": 0.0},
    }

    try:
        import cadquery as cq

        shape = cq.importers.importStep(step_path).val()
        bb = shape.BoundingBox()
        result["bounding_box_mm"] = {
            "x": round(bb.xmax - bb.xmin, 3),
            "y": round(bb.ymax - bb.ymin, 3),
            "z": round(bb.zmax - bb.zmin, 3),
        }

        for face in shape.Faces():
            geom_type = face.geomType()
            fbb = face.BoundingBox()

            if geom_type == "CYLINDER":
                spans = sorted(
                    [
                        fbb.xmax - fbb.xmin,
                        fbb.ymax - fbb.ymin,
                        fbb.zmax - fbb.zmin,
                    ]
                )
                diameter = round((spans[0] + spans[1]) / 2.0, 2)
                depth = round(spans[2], 2)
                if diameter > 0.5 and depth > 0.5:
                    result["holes"].append(
                        {"diameter_mm": diameter, "depth_mm": depth}
                    )

            elif geom_type == "PLANE":
                width = round(fbb.xmax - fbb.xmin, 2)
                length = round(fbb.ymax - fbb.ymin, 2)
                depth = round(fbb.zmax - fbb.zmin, 2)
                bb_x = result["bounding_box_mm"]["x"]
                bb_y = result["bounding_box_mm"]["y"]
                bb_z = result["bounding_box_mm"]["z"]

                if depth < 0.01:
                    planar_width = width
                    planar_length = length
                    planar_depth = round(bb.zmax - fbb.zmax, 2)
                else:
                    spans = sorted([width, length, depth])
                    planar_depth = spans[0]
                    planar_width = spans[1]
                    planar_length = spans[2]

                if (
                    planar_depth > 0.5
                    and planar_width < max(bb_x, bb_y, bb_z) * 0.95
                    and planar_length < max(bb_x, bb_y, bb_z) * 0.95
                    and planar_width > 3.0
                    and planar_length > 3.0
                ):
                    result["planar_recesses"].append(
                        {
                            "width_mm": planar_width,
                            "length_mm": planar_length,
                            "depth_mm": max(planar_depth, 1.0),
                        }
                    )

    except Exception as exc:
        result["warnings"] = [f"B-rep measurement error: {exc}"]

    return result


def assemble_feature_attributes(
    detected_features: list[dict],
    step_pmi: dict,
    brep_data: dict,
    default_material: str = "aluminium_6061",
) -> dict:
    """Match detected feature classes to PMI and measured geometry."""
    material = step_pmi.get("material") or default_material
    material_source = "pmi" if step_pmi.get("material") else "default"

    global_ra = None
    global_ra_source = "default"
    if step_pmi.get("surface_finish"):
        values = [sf["Ra_um"] for sf in step_pmi["surface_finish"]]
        global_ra = round(sum(values) / len(values), 2)
        global_ra_source = "pmi"

    holes = sorted(
        brep_data.get("holes", []), key=lambda hole: hole["diameter_mm"], reverse=True
    )
    recesses = sorted(
        brep_data.get("planar_recesses", []),
        key=lambda recess: recess["width_mm"] * recess["length_mm"],
        reverse=True,
    )
    thread_queue = [thread["spec"] for thread in step_pmi.get("threads", [])]
    hole_queue = list(holes)
    recess_queue = list(recesses)
    hole_types = {"through_hole", "blind_hole"}
    recess_types = {
        "rectangular_pocket",
        "circular_pocket",
        "rectangular_slot",
        "circular_slot",
        "rectangular_step",
        "triangular_pocket",
    }

    features_out = []
    instance_counters: dict[str, int] = {}

    for feature in detected_features:
        feature_type = feature["type"]
        instance_id = instance_counters.get(feature_type, 0)
        instance_counters[feature_type] = instance_id + 1
        ra_um = global_ra or DEFAULT_RA.get(feature_type, 3.2)

        entry = {
            "type": feature_type,
            "instance_id": instance_id,
            "Ra_um": ra_um,
            "Ra_source": global_ra_source if global_ra else "default",
            "threaded": False,
            "thread_spec": None,
        }

        if feature_type in hole_types:
            if hole_queue:
                hole = hole_queue.pop(0)
                diameter = hole["diameter_mm"]
                depth = hole["depth_mm"]
            else:
                bb = brep_data.get("bounding_box_mm", {})
                diameter = round(min(bb.get("x", 20), bb.get("y", 20)) * 0.15, 1)
                depth = round(bb.get("z", 40) * 0.6, 1)
            entry["diameter_mm"] = diameter
            entry["depth_mm"] = depth
            entry["depth_ratio"] = round(depth / max(diameter, 0.1), 2)
            entry["peck_required"] = entry["depth_ratio"] > PECK_DEPTH_RATIO
            if thread_queue:
                entry["threaded"] = True
                entry["thread_spec"] = thread_queue.pop(0)

        elif feature_type in recess_types:
            if recess_queue:
                recess = recess_queue.pop(0)
                entry["width_mm"] = recess["width_mm"]
                entry["length_mm"] = recess["length_mm"]
                entry["depth_mm"] = recess["depth_mm"]
                doc = ROUGH_DOC_MM.get(material, 2.0)
                entry["rough_passes"] = max(1, int((recess["depth_mm"] / doc) + 0.5))
            else:
                bb = brep_data.get("bounding_box_mm", {})
                entry["width_mm"] = round(bb.get("x", 50) * 0.4, 1)
                entry["length_mm"] = round(bb.get("y", 40) * 0.4, 1)
                entry["depth_mm"] = round(bb.get("z", 30) * 0.3, 1)
                entry["rough_passes"] = 2

        features_out.append(entry)

    return {
        "material": material,
        "material_source": material_source,
        "features": features_out,
    }


def extract_pmi(
    step_path: str,
    features_path: str,
    output_dir: str,
    default_material: str = "aluminium_6061",
) -> dict:
    """Extract explicit and implicit PMI from a STEP file into pmi_data.json."""
    if not os.path.exists(step_path):
        raise FileNotFoundError(step_path)
    if not os.path.exists(features_path):
        raise FileNotFoundError(features_path)

    with open(features_path, encoding="utf-8") as f:
        features_json = json.load(f)
    detected_features = features_json.get("features", [])
    if not isinstance(detected_features, list):
        raise ValueError("features JSON must contain a 'features' list.")

    step_pmi = parse_step_pmi(step_path)
    brep_data = measure_brep_features(step_path)
    assembled = assemble_feature_attributes(
        detected_features, step_pmi, brep_data, default_material
    )

    output_abs = os.path.abspath(output_dir)
    pmi_data_file = os.path.join(output_abs, "pmi_data.json")
    warnings = list(brep_data.get("warnings", []))
    if assembled["material_source"] == "pmi" and assembled["material"] != default_material:
        warnings.append(
            f"PMI material '{assembled['material']}' overrides requested material '{default_material}'."
        )

    result = {
        "source_file": os.path.abspath(step_path),
        "material": assembled["material"],
        "material_source": assembled["material_source"],
        "features": assembled["features"],
        "pmi_data_file": os.path.abspath(pmi_data_file),
        "warnings": warnings,
    }
    _write_json_atomic(result, pmi_data_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PMI from a STEP file.")
    parser.add_argument("step_path")
    parser.add_argument("features_path")
    parser.add_argument("output_dir")
    parser.add_argument("--material", default="aluminium_6061")
    args = parser.parse_args()

    result = extract_pmi(
        args.step_path,
        args.features_path,
        args.output_dir,
        default_material=args.material,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
