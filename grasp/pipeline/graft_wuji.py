"""Graft the Wuji hand onto the six-axis wrists of an existing OakInk2 task scene (Sharpa source).

Keeps the source objects, connections, supports and contact parameters, and the Wuji meshes, inertias, joint limits
and finger PD/torque limits (robot description from TopoRetarget, https://github.com/wuji-technology/toporetarget,
``robots/wujihand2_{right,left}.json``).  Collision geoms are renamed per finger segment so each Sharpa contact pair
maps onto the matching Wuji geoms; a palm site in the Sharpa/XHand palm frame (+x palm normal, +z to the fingers) and
fingertip sites are added for IK and tracking; servos are explicit general actuators so no Sharpa defaults leak in.
The compiled model is checked against the raw Wuji model and the source objects.  The IK step
(``grasp.pipeline.solve_ik --robot-type wuji_topo``) then reads the task's MANO keypoints.

    python -m grasp.pipeline.graft_wuji --task oakbi_000_cut --source-dir data/oakink2/processed/oakink \
        --dataset-dir data/oakink2_wuji --wuji-config-dir ~/toporetarget/toporetarget/robots
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def graft_wuji(task: str, source_dir: Path, dataset_dir: Path, wuji_config_dir: Path, robot_type: str = 'wuji_topo'):
    source = source_dir / 'sharpa_bimanual/bimanual' / task
    dest = dataset_dir / 'processed/oakink' / robot_type / 'bimanual' / task
    (dest / '0').mkdir(parents=True, exist_ok=True)
    root = ET.parse(source / 'scene.xml').getroot()
    original = mujoco.MjModel.from_xml_path(str(source / 'scene.xml'))
    compiler = root.find('compiler')
    compiler.set('meshdir', str(source_dir / 'assets'))
    world, asset, actuator, contact = [root.find(x) for x in ('worldbody', 'asset', 'actuator', 'contact')]
    # Graft onto each original six-axis wrist, preserving its dynamics/servos.
    old_geoms, old_bodies, old_joints = set(), set(), set()
    new_geoms = {}
    geom_map = {}
    provenance = {}
    for side in ('right', 'left'):
        config_path = wuji_config_dir / f'wujihand2_{side}.json'
        config = json.loads(config_path.read_text())['robot']
        xml_path = (config_path.parent / config['xml']).resolve()
        hand = ET.parse(xml_path).getroot()
        raw_model = mujoco.MjModel.from_xml_path(str(xml_path))
        provenance[side] = dict(xml=str(xml_path), sha256=hashlib.sha256(xml_path.read_bytes()).hexdigest())
        yaw = world.find(f".//body[@name='{side}_base_yaw']")
        assert yaw is not None
        old = next(x for x in yaw if x.tag == 'body')
        old_geoms.update(g.get('name') for g in old.iter('geom') if g.get('name'))
        old_bodies.update(g.get('name') for g in old.iter('body'))
        old_joints.update(g.get('name') for g in old.iter('joint'))
        yaw.remove(old)
        new = hand.find('worldbody/body')
        prefix = side[0] + '_'
        mats = {m.get('name'): side + '_' + m.get('name') for m in hand.findall('asset/material')}
        for a in hand.find('asset'):
            if a.tag == 'mesh':
                a.set('file', str((xml_path.parent / hand.find('compiler').get('meshdir', '') / a.get('file')).resolve()))
            if a.tag == 'material':
                a.set('name', mats[a.get('name')])
            asset.append(a)
        # Explicit inheritance avoids Sharpa's armature=1 leaking into Wuji.
        for j in new.iter('joint'):
            j.set('armature', j.get('armature', '0.0002'))
            j.set('damping', j.get('damping', '0'))
            j.set('frictionloss', j.get('frictionloss', '0'))
        new_geoms[side] = []
        for body in new.iter('body'):
            for i, g in enumerate(body.findall('geom')):
                if g.get('material'):
                    g.set('material', mats[g.get('material')])
                collision = g.get('contype', '1') != '0' or g.get('conaffinity', '1') != '0'
                if collision:
                    g.set('name', f'collision_hand_{side}_{body.get("name")}_{i}')
                    g.set('group', '3')
                    new_geoms[side].append(g.get('name'))
                    bn = body.get('name')
                    finger = next((f for f in FINGERS if bn.startswith(prefix + f + '_')), 'palm')
                    segment = (0 if 'distal' in bn or 'tip_sensor' in bn else
                               1 if bn.endswith('_middle') and finger != 'palm' else 2)
                    if finger == 'palm':
                        segment = 0
                    geom_map.setdefault(f'collision_hand_{side}_{finger}_{segment}', []).append(g.get('name'))
                else:
                    g.set('name', f'visual_{body.get("name")}_{i}')
                g.set('contype', '0'); g.set('conaffinity', '0')
        for f, definition in zip(FINGERS, config['hand']['fingers']):
            site = new.find(f".//site[@name='{definition['tip_site']}']")
            assert site is not None
            site.set('name', f'{side}_{f}_tip')
            parent = next(b for b in new.iter('body') if site in list(b))
            for pre in ('track_hand_', 'trace_hand_'):
                duplicate = copy.deepcopy(site)
                duplicate.set('name', f'{pre}{side}_{f}_tip')
                parent.append(duplicate)
        # Canonical palm frame: +x palm normal, +y across palm, +z to fingers.
        # Wuji has +y palm normal, +x across palm, -z to fingers.
        wrist = new.find(f".//body[@name='{prefix}wrist']")
        # 2026-10-29 fix: the left Wuji hand is the mirror image; its palm site is the right one turned 180 deg about
        # the finger axis, so +x stays the palm normal and the thumb sits on -y (as in the Sharpa/XHand sites).
        quat = '0 0.7071067811865476 0.7071067811865476 0' if side == 'right' else '0 0.7071067811865476 -0.7071067811865476 0'
        ET.SubElement(wrist, 'site', name=f'{side}_palm', pos='0 0 -0.04', quat=quat, size='0.005')
        yaw.append(new)
        # Use explicit general servos to avoid inherited Sharpa gain defaults.
        for a in hand.find('actuator'):
            aa = dict(a.attrib)
            aa['name'] = aa['joint'] + '_position'
            kp, kv = aa.pop('kp'), aa.pop('kv')
            aa.update(gaintype='fixed', biastype='affine', gainprm=kp, biasprm=f'0 -{kp} -{kv}',
                      ctrllimited='true', forcelimited='true')
            actuator.append(ET.Element('general', aa))
        provenance[side]['collision_geoms'] = len(new_geoms[side])
        assert raw_model.nu == 20 and len(new_geoms[side]) == 27
    for a in list(actuator):
        if a.get('joint') in old_joints:
            actuator.remove(a)
    # Order controls exactly like joint qpos: wrist + fingers, right then left.
    joint_order = {j.get('name'): i for i, j in enumerate(world.iter('joint'))}
    actuator[:] = sorted(actuator, key=lambda a: joint_order[a.get('joint')])
    for p in list(contact):
        if p.tag == 'exclude':
            if p.get('body1') in old_bodies or p.get('body2') in old_bodies:
                contact.remove(p)
            continue
        g1, g2 = p.get('geom1'), p.get('geom2')
        if g1 in old_geoms or g2 in old_geoms:
            attrs = {k: v for k, v in p.attrib.items() if k not in ('name', 'geom1', 'geom2')}
            aa = geom_map[g1] if g1 in old_geoms else [g1]
            bb = geom_map[g2] if g2 in old_geoms else [g2]
            for a in aa:
                for b in bb:
                    ET.SubElement(contact, 'pair', geom1=a, geom2=b, name=f'{a}__{b}', **attrs)
            contact.remove(p)
    # Drop unused Sharpa meshes, retaining every object/support asset.
    used_meshes = {g.get('mesh') for g in world.iter('geom')}
    for a in list(asset):
        if a.tag == 'mesh' and a.get('name') not in used_meshes:
            asset.remove(a)
    for tag in ('keyframe',):
        for element in root.findall(tag):
            root.remove(element)
    for eq in root.findall('equality/*'):
        assert not any(v in old_bodies or v in old_joints for v in eq.attrib.values())
    ET.indent(root)
    for path in (dest / 'scene.xml', dest / '0/scene_ik.xml'):
        ET.ElementTree(root).write(path)
    model = mujoco.MjModel.from_xml_path(str(dest / 'scene.xml'))
    assert (model.nq, model.nv, model.nu) == (original.nq-4, original.nv-4, original.nu-4)
    # Verify actual compiled dynamics against upstream, not just XML text.
    for side in ('right', 'left'):
        raw = mujoco.MjModel.from_xml_path(provenance[side]['xml'])
        for aid in range(raw.nu):
            name = raw.joint(raw.actuator_trnid[aid, 0]).name + '_position'
            other = model.actuator(name).id
            for attr in ('actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange'):
                np.testing.assert_allclose(getattr(model, attr)[other], getattr(raw, attr)[aid])
        for jid in range(raw.njnt):
            other = model.joint(raw.joint(jid).name).id
            np.testing.assert_allclose(model.jnt_range[other], raw.jnt_range[jid])
            np.testing.assert_allclose(model.dof_armature[model.jnt_dofadr[other]], raw.dof_armature[raw.jnt_dofadr[jid]])
        for bid in range(1, raw.nbody):
            other = model.body(raw.body(bid).name).id
            for attr in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat'):
                np.testing.assert_allclose(getattr(model, attr)[other], getattr(raw, attr)[bid])
    for name in ('right_object', 'left_object'):
        a, b = original.body(name).id, model.body(name).id
        for attr in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat'):
            np.testing.assert_allclose(getattr(original, attr)[a], getattr(model, attr)[b])
    mano = dataset_dir / 'processed/oakink/mano/bimanual' / task
    source_mano = source_dir / 'mano/bimanual' / task
    if mano.resolve() != source_mano.resolve() and not mano.exists():
        mano.parent.mkdir(parents=True, exist_ok=True)
        mano.symlink_to(source_mano, target_is_directory=True)
    info = json.loads((source / 'task_info.json').read_text())
    info.update(robot_type=robot_type, dataset_dir=str(dataset_dir))
    (dest / 'task_info.json').write_text(json.dumps(info, indent=2) + '\n')
    return model, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True)
    parser.add_argument('--source-dir', type=Path, required=True, help='processed/oakink of the Sharpa dataset')
    parser.add_argument('--dataset-dir', type=Path, required=True, help='output dataset directory')
    parser.add_argument('--wuji-config-dir', type=Path, required=True, help='TopoRetarget robots/ directory')
    parser.add_argument('--robot-type', default='wuji_topo')
    args = parser.parse_args()
    model, _ = graft_wuji(args.task, args.source_dir.resolve(), args.dataset_dir.resolve(),
                          args.wuji_config_dir.resolve(), args.robot_type)
    print(f'[{args.task}] {args.robot_type} scene: nq {model.nq}, nu {model.nu}, contact pairs {model.npair}', flush=True)


if __name__ == '__main__':
    main()
