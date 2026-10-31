"""Put another bimanual hand into an existing OakInk2 task scene, keeping the objects, supports and contact policy.

The source scene is a processed task of one robot (e.g. ``sharpa_bimanual``); the new robot is a directory with
``right.xml``, ``left.xml`` and ``bimanual.xml`` whose hand bodies carry ``right_palm`` / ``left_palm`` sites.  The old
hand bodies, their actuators, excludes and contact pairs are removed; pairs between geoms that exist in both hands are
kept.  The result is checked against the native robot model (gains, joints, inertias, collision shapes) and against
the source scene (every non-robot body unchanged).  The IK step (``grasp.pipeline.solve_ik``) then reads the task's
MANO keypoints from the same dataset directory.

    python -m grasp.pipeline.swap_robot --task oakbi_000_cut --dataset-dir data/oakink2 \\
        --source-robot sharpa_bimanual --robot-dir data/oakink2/processed/oakink/assets/robots/xhand --robot-type xhand
"""

from __future__ import annotations

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from grasp.pipeline.generate_xml import SUPPORT_PLATE_FLOOR_ONLY_TASKS, _support_plates_floor_only


def swap_robot(source_scene: Path, robot_dir: Path, robot_type: str, dest: Path) -> mujoco.MjModel:
    source_scene = source_scene.resolve()      # the scene's relative meshdir is relative to its real location
    root = ET.parse(source_scene).getroot()
    original = mujoco.MjModel.from_xml_path(str(source_scene))
    (dest / "0").mkdir(parents=True, exist_ok=True)
    compiler = root.find("compiler")
    compiler.set("meshdir", str((source_scene.parent / compiler.get("meshdir", "")).resolve()))
    world, asset, actuator, contact = [root.find(k) for k in ("worldbody", "asset", "actuator", "contact")]
    old_bodies, old_joints, old_geoms, new_geoms = set(), set(), set(), set()
    native = mujoco.MjModel.from_xml_path(str(robot_dir / "bimanual.xml"))
    for side in ("right", "left"):
        hand = ET.parse(robot_dir / f"{side}.xml").getroot()
        old = next(b for b in world.findall("body") if b.find(f".//site[@name='{side}_palm']") is not None)
        old_bodies.update(b.get("name") for b in old.iter("body"))
        old_joints.update(j.get("name") for j in old.iter("joint"))
        old_geoms.update(g.get("name") for g in old.iter("geom") if g.get("name"))
        index = list(world).index(old)
        world.remove(old)
        defaults = copy.deepcopy(hand.find("default"))
        defaults.set("class", f"{robot_type}_{side}")
        root.find("default").append(defaults)
        for body in hand.find("worldbody"):
            body.set("childclass", f"{robot_type}_{side}")
            new_geoms.update(g.get("name") for g in body.iter("geom") if g.get("name"))
            world.insert(index, body)
            index += 1
        for a in hand.find("asset"):
            if a.tag == "mesh":
                a.set("file", str((robot_dir / hand.find("compiler").get("meshdir", "") / a.get("file")).resolve()))
                asset.append(a)
        for a in hand.find("actuator"):
            a.set("class", f"{robot_type}_{side}")
            actuator.append(a)
    for a in list(actuator):
        if a.get("joint") in old_joints:
            actuator.remove(a)
    # keep the source contact policy for collision geoms present in the new hand
    for pair in list(contact):
        if pair.tag == "exclude":
            if pair.get("body1") in old_bodies or pair.get("body2") in old_bodies:
                contact.remove(pair)
        elif any(pair.get(k) in old_geoms and pair.get(k) not in new_geoms for k in ("geom1", "geom2")):
            contact.remove(pair)
    used = {g.get("mesh") for g in world.iter("geom")}
    for a in list(asset):
        if a.tag == "mesh" and a.get("name") not in used:
            asset.remove(a)
    for e in root.findall("keyframe"):
        root.remove(e)
    for e in root.findall("equality/*"):
        assert not any(v in old_bodies or v in old_joints for v in e.attrib.values())
    ET.indent(root)
    for f in (dest / "scene.xml", dest / "0/scene_ik.xml"):
        ET.ElementTree(root).write(f)
    model = mujoco.MjModel.from_xml_path(str(dest / "scene.xml"))

    # the robot is the native model: gains, ranges, joints, inertias and collision shapes
    for aid in range(native.nu):
        other = model.actuator(native.actuator(aid).name).id
        for attr in ("actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange"):
            np.testing.assert_allclose(getattr(model, attr)[other], getattr(native, attr)[aid], atol=1e-8)
    for jid in range(native.njnt):
        other = model.joint(native.joint(jid).name).id
        for attr in ("jnt_range", "jnt_axis", "jnt_pos"):
            np.testing.assert_allclose(getattr(model, attr)[other], getattr(native, attr)[jid])
        for attr in ("dof_armature", "dof_damping", "dof_frictionloss"):
            np.testing.assert_allclose(getattr(model, attr)[model.jnt_dofadr[other]],
                                       getattr(native, attr)[native.jnt_dofadr[jid]])
    for bid in range(1, native.nbody):
        other = model.body(native.body(bid).name).id
        for attr in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
            np.testing.assert_allclose(getattr(model, attr)[other], getattr(native, attr)[bid], atol=1e-8)
    for gid in range(native.ngeom):
        name = native.geom(gid).name
        if not name.startswith("collision_hand_"):
            continue
        other = model.geom(name).id
        for attr in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction"):
            np.testing.assert_allclose(getattr(model, attr)[other], getattr(native, attr)[gid], atol=1e-8)
    # every non-robot body (objects, connections, supports) is the source scene's
    for bid in range(1, original.nbody):
        name = original.body(bid).name
        if name in old_bodies:
            continue
        other = model.body(name).id
        for attr in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
            np.testing.assert_allclose(getattr(model, attr)[other], getattr(original, attr)[bid], atol=1e-8)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-robot", default="sharpa_bimanual")
    parser.add_argument("--robot-dir", type=Path, required=True, help="directory with right.xml, left.xml, bimanual.xml")
    parser.add_argument("--robot-type", required=True)
    args = parser.parse_args()
    processed = args.dataset_dir.resolve() / "processed/oakink"
    source = processed / args.source_robot / "bimanual" / args.task
    dest = processed / args.robot_type / "bimanual" / args.task
    # start from the source robot's scene before its own support-plate change, if it has one
    before = source / "scene.xml.before_floor_only"
    model = swap_robot(before if before.exists() else source / "scene.xml", args.robot_dir.resolve(), args.robot_type, dest)
    info = json.loads((source / "task_info.json").read_text())
    if args.task in SUPPORT_PLATE_FLOOR_ONLY_TASKS.get(args.robot_type, ()):
        xml = (dest / "scene.xml").read_text()
        (dest / "scene.xml.before_floor_only").write_text(xml)
        (dest / "scene.xml").write_text(_support_plates_floor_only(xml, model, info))
        model = mujoco.MjModel.from_xml_path(str(dest / "scene.xml"))
    info.update(robot_type=args.robot_type, dataset_dir=str(args.dataset_dir.resolve()))
    (dest / "task_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"[{args.task}] {args.robot_type} scene {dest / 'scene.xml'}: nq {model.nq}, nv {model.nv}, nu {model.nu}",
          flush=True)


if __name__ == "__main__":
    main()
