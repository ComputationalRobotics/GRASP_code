"""Load object-local physical connections without changing IK or references."""

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def add_object_connections(xml_text, task_info, config_path=None):
    """Add a configured hinge to the runtime XML; unmatched objects are unchanged.

    Two distinct connect anchors define a hinge while preserving the existing
    free-joint qpos layout. Coordinates are in each object's mesh/body frame.
    This function deliberately does not modify the IK scene or reference data.
    """
    if config_path is None:
        return xml_text
    config = json.loads(Path(config_path).read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported object connection schema")
    objects = {
        side: Path(task_info.get(f"{side}_object_mesh_dir") or "").name.split("__")[0]
        for side in ("right", "left")
    }
    pair = "|".join(sorted(objects.values()))
    connection = config["connections"].get(pair)
    if connection is None:
        return xml_text
    if connection["type"] != "hinge" or len(set(objects.values())) != 2:
        raise ValueError("A hinge requires two distinct object parts")
    span = float(connection["anchor_spacing_m"])
    solref = [float(x) for x in connection["solref"]]
    if not math.isfinite(span) or span <= 0 or len(solref) != 2 or not all(
        math.isfinite(x) and x > 0 for x in solref
    ):
        raise ValueError("Invalid hinge spacing or solver parameters")
    root = ET.fromstring(xml_text)
    expected = []
    for side, object_id in objects.items():
        body = root.find(f".//worldbody//body[@name='{side}_object']")
        if body is None:
            raise ValueError(f"Missing {side}_object body")
        part = connection["parts"][object_id]
        point, axis = part["point_m"], part["axis"]
        if len(point) != 3 or len(axis) != 3 or not all(
            math.isfinite(x) for x in [*point, *axis]
        ) or abs(sum(x * x for x in axis) - 1) > 1e-6:
            raise ValueError(f"Invalid hinge geometry for {object_id}")
        for j in range(2):
            attrs = dict(
                name=f"{side}_part_hinge_{j}",
                pos=" ".join(f"{p + j * span * a:.12g}" for p, a in zip(point, axis)),
                size=".001", rgba="0 0 0 0",
            )
            expected.append((body, "site", attrs))
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for j in range(2):
        expected.append((equality, "connect", dict(
            name=f"part_hinge_{j}", site1=f"right_part_hinge_{j}",
            site2=f"left_part_hinge_{j}", solref=" ".join(str(x) for x in solref),
        )))
    changed = False
    for parent, tag, attrs in expected:
        matches = root.findall(f".//{tag}[@name='{attrs['name']}']")
        if matches:
            if len(matches) != 1 or matches[0] not in list(parent) or matches[0].attrib != attrs:
                raise ValueError(f"Conflicting object connection: {attrs['name']}")
        else:
            ET.SubElement(parent, tag, attrs)
            changed = True
    return ET.tostring(root, encoding="unicode") if changed else xml_text
