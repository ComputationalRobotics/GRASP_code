"""Post-IK support surfaces for HOT3D objects resting away from the floor.

Uses the actual collision geometry and IK endpoint poses. No trajectory, robot,
object mass, or controller parameters are changed. Supports are static surfaces,
not geometry welded to the manipulated object.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

VERSION = 'endpoint_support_v1'
HAND_NEAR_M = .03
FLOOR_NEAR_M = .005


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(task_dir):
    task_dir = Path(task_dir)
    source = task_dir / '0/scene_ik.xml'
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    with np.load(task_dir / '0/trajectory_kinematic.npz') as z:
        qpos = z['qpos'].copy()
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '' for i in range(model.ngeom)]
    hands = [i for i, n in enumerate(names) if n.startswith('collision_hand_')]
    floor = names.index('floor')
    records = []
    for endpoint, frame in [('start', 0), ('end', -1)]:
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        floor_z = float(data.geom_xpos[floor, 2])
        for side in ['right', 'left']:
            ids = [i for i, n in enumerate(names) if n.startswith(side + '_object_')
                   and model.geom_group[i] == 3 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
            if not ids:
                continue
            parts = []
            for i in ids:
                mesh = model.geom_dataid[i]
                a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                parts.append(model.mesh_vert[a:a+n] @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i])
            verts = np.concatenate(parts)
            bottom = float(verts[:, 2].min())
            distance = min(float(mujoco.mj_geomDistance(model, data, h, o, 10., None)) for h in hands for o in ids)
            lo, hi = verts[:, :2].min(0), verts[:, :2].max(0)
            center, half = (lo + hi) / 2, (hi - lo) / 2 + .005
            records.append(dict(side=side, endpoint=endpoint, bottom_z=bottom,
                                floor_gap_m=bottom-floor_z, hand_distance_m=distance,
                                needs_support=bool(bottom-floor_z > FLOOR_NEAR_M and distance >= HAND_NEAR_M),
                                center_xy=center.tolist(), halfsize_xy=half.tolist(),
                                object_geoms=[names[i] for i in ids]))
    return dict(version=VERSION, task=task_dir.name, endpoints=records,
                reference_sha256=digest(task_dir / '0/trajectory_kinematic.npz'),
                ik_scene_sha256=digest(source), dimensions=[model.nq, model.nv, model.nu])


def resolve(task_dir):
    task_dir = Path(task_dir)
    marker = task_dir / 'support_audit.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if (old.get('version') == VERSION and old['reference_sha256'] == digest(task_dir / '0/trajectory_kinematic.npz')
                and old['ik_scene_sha256'] == digest(task_dir / '0/scene_ik.xml')
                and old.get('scene_sha256') == digest(task_dir / 'scene.xml')):
            return old
    audit = inspect(task_dir)
    placements = []
    for r in audit['endpoints']:
        if not r['needs_support']:
            continue
        if any(abs(r['bottom_z'] - p['bottom_z']) < .01 and
               np.all(np.abs(np.array(r['center_xy'])-p['center_xy']) < np.minimum(r['halfsize_xy'], p['halfsize_xy']))
               for p in placements):
            continue
        placements.append(r)
    # Start from the untouched structural scenes on every invocation.
    for name in ['scene.xml', 'scene_eq.xml']:
        path = task_dir / name
        backup = task_dir / (name + '.before_support')
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        tree = ET.parse(backup)
        root = tree.getroot()
        world, contact = root.find('worldbody'), root.find('contact')
        for p in placements:
            support = f"hot3d_support_{p['side']}_{p['endpoint']}"
            ET.SubElement(world, 'geom', name=support, type='box',
                          pos=' '.join(map(str, p['center_xy'] + [p['bottom_z']-.005-.0001])),
                          size=' '.join(map(str, p['halfsize_xy'] + [.005])),
                          contype='0', conaffinity='0', rgba='.3 .4 .5 1')
            # Use the existing floor contact parameters for object/support pairs.
            for pair in list(contact):
                if pair.get('geom1') != 'floor':
                    continue
                attrs = dict(pair.attrib)
                attrs.update(geom1=support, name=support + '_' + attrs['geom2'])
                ET.SubElement(contact, 'pair', attrs)
            for g in root.findall('.//geom'):
                hand = g.get('name', '')
                if hand.startswith('collision_hand_'):
                    ET.SubElement(contact, 'pair', name=support+'_'+hand, geom1=support, geom2=hand,
                                  condim='3', friction='1 1 0.1 0 0')
        if placements:
            temp = path.with_suffix('.support_tmp.xml')
            tree.write(temp)
            check = mujoco.MjModel.from_xml_path(str(temp))
            assert [check.nq, check.nv, check.nu] == audit['dimensions']
            temp.replace(path)
        else:
            # Retain byte-identical scenes and completed results when unaffected.
            path.write_bytes(backup.read_bytes())
    audit['placements'] = placements
    audit['scene_sha256'] = digest(task_dir / 'scene.xml')
    assert audit['reference_sha256'] == digest(task_dir / '0/trajectory_kinematic.npz')
    assert audit['ik_scene_sha256'] == digest(task_dir / '0/scene_ik.xml')
    marker.write_text(json.dumps(audit, indent=2) + '\n')
    return audit
