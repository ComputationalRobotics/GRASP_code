# Data Preparation

Run these commands from the repository root after installing `.[pipeline]`. Models are bundled under `grasp/assets/robots`; input demonstrations and object meshes are separate data assets.

## OakInk2: Sharpa

The preparation entry point consumes one bimanual pickle and its object meshes:

```text
data/oakink2/
└── raw/oakink/
    ├── oakbi_443_take_outside_bimanual.pkl
    └── meshes/
        └── C43001/gamecontroller.ply
```

These are inputs, not files bundled with the repository. The mesh paths under `meshes/` must match the object paths recorded in the pickle after `align_ds/`.

Copy the unmodified Sharpa model to the dataset's robot asset directory, then prepare the task:

```bash
mkdir -p data/oakink2/processed/oakink/assets/robots/sharpa_bimanual
cp -a grasp/assets/robots/sharpa/. \
  data/oakink2/processed/oakink/assets/robots/sharpa_bimanual/

python -m grasp.pipeline.prepare_oakink2 \
  --task oakbi_443_take_outside \
  --dataset-dir data/oakink2 \
  --robot-type sharpa_bimanual
```

This runs keypoint conversion, object decomposition, scene generation, and IK. The resulting task directory is:

```text
data/oakink2/processed/oakink/sharpa_bimanual/bimanual/oakbi_443_take_outside/
├── scene.xml
├── task_info.json
└── 0/trajectory_kinematic.npz
```

Run physics optimization with the [README command](../README.md#run-grasp-mpc).

### Exporting a pickle from original OakInk2 data

If starting from the original release, [`oakink2_maniptrans.py`](../grasp/pipeline/oakink2_maniptrans.py) uses the loaders in an external [ManipTrans checkout](https://github.com/ManipTrans/ManipTrans). Set up its OakInk2 data under `<maniptrans-root>/data/OakInk-v2`, then supply the source clip's ManipTrans data index:

```bash
python -m grasp.pipeline.oakink2_maniptrans 'HASH_PREFIX@PRIMITIVE_INDEX' \
  --task YOUR_TASK \
  --dataset-dir data/oakink2 \
  --maniptrans-root /path/to/ManipTrans \
  --oakink2-root /path/to/OakInk-v2 \
  --target-fps 50 --max-seconds 20
```

Replace the index and task placeholders with the chosen clip. ManipTrans is needed for this export step. Starting from an exported pickle and meshes, scene preparation, IK, MPC, and evaluation use this repository and its declared Python dependencies.

## OakInk2: XHand and Wuji

Prepare the Sharpa source scene above first. The following commands keep its objects and build the selected hand scene from the bundled robot model.

**XHand:**

```bash
python -m grasp.pipeline.swap_robot \
  --task oakbi_443_take_outside \
  --dataset-dir data/oakink2 \
  --source-robot sharpa_bimanual \
  --robot-dir grasp/assets/robots/xhand \
  --robot-type xhand
```

**Wuji:**

```bash
python -m grasp.pipeline.graft_wuji \
  --task oakbi_443_take_outside \
  --source-dir data/oakink2/processed/oakink \
  --dataset-dir data/oakink2 \
  --wuji-config-dir grasp/assets/robots/wuji/robots
```

Solve IK for the selected hand. This example uses XHand; substitute `wuji_topo` for Wuji. Read the frame count from the keypoint file so the full reference is retained:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from grasp.pipeline.solve_ik import main

root = Path("data/oakink2/processed/oakink").resolve()
task = "oakbi_443_take_outside"
keypoints = np.load(root / "mano/bimanual" / task / "0/trajectory_keypoints.npz")
main(output_root_dir=str(root), robot_type="xhand", task=task,
     end_idx=len(keypoints["qpos_wrist_right"]), ref_dt=0.02, sim_dt=0.005,
     smoothing=False, show_viewer=False, save_video=False)
PY
```

Run the prepared hand with `experiments/paper.py oakink2 --robot xhand` or `--robot wuji_topo`, specifying the task and dataset directory as in the README.

The original XHand and Wuji evaluation batches used an EPA horizon of 64. To reproduce that setting with the pinned MuJoCo Warp version:

```bash
python - oakink2 --robot xhand --task oakbi_443_take_outside \
  --dataset-dir data/oakink2 -- --output-dir runs/xhand_443 <<'PY'
from mujoco_warp._src import collision_convex
collision_convex.MJ_MAX_EPAHORIZON = 64
import runpy
runpy.run_path("experiments/paper.py", run_name="__main__")
PY
```

Use `--robot wuji_topo` and a separate output directory for Wuji. Set `JAX_PLATFORMS=cpu` as in the installation instructions.

## HOT3D

Supply the MANO model directory containing the left and right hand models. The exporter downloads the selected clip's annotations and object models when they are absent locally.

```bash
mkdir -p data/hot3d/processed/hot3d/assets/robots/sharpa
cp -a grasp/assets/robots/sharpa/. \
  data/hot3d/processed/hot3d/assets/robots/sharpa/

python -m grasp.pipeline.hot3d export \
  --split train_aria --clip clip-001849 \
  --dataset-dir data/hot3d --mano-dir /path/to/mano/models

python -m grasp.pipeline.hot3d prepare \
  --split train_aria --clip clip-001849 --dataset-dir data/hot3d

python experiments/paper.py hot3d \
  --task hot3d_clip-001849_obj28 --dataset-dir data/hot3d \
  -- --output-dir runs/hot3d_001849
```

The exported `raw/<split>/<clip>/tasks.json` lists each task and its embodiment. For a single-hand task, pass `--embodiment right` or `left` to the MPC launcher.

## Humanoid References

### LAFAN1

The converter downloads the selected G1 reference from the LAFAN1 retargeting dataset. Use `--csv /path/to/clip.csv` to supply it locally.

```bash
python -m grasp.pipeline.lafan \
  --task dance1_subject1 --dataset-dir data/lafan \
  --robot-asset-dir grasp/assets/robots/unitree_g1

python experiments/paper.py lafan \
  --task dance1_subject1 --dataset-dir data/lafan \
  -- --output-dir runs/lafan_dance
```

### AMASS

Start from SPIDER's released G1 trajectories, with `--source-dir` pointing to its `processed/amass` directory:

```bash
python -m grasp.pipeline.amass \
  --task martial_arts --source-dir /path/to/processed/amass \
  --dataset-dir data/amass \
  --robot-asset-dir grasp/assets/robots/unitree_g1

python experiments/paper.py amass \
  --task martial_arts --dataset-dir data/amass \
  -- --output-dir runs/amass_martial_arts
```

Use `--source-initial-velocity` during preparation for the paper's `cartwheels`, `fight_and_sports`, `run_to_crouch`, and `walk_to_leap_to_walk` tasks.

### OMOMO

The MPC launcher accepts prepared G1 object-interaction tasks under `processed/omomo/unitree_g1/humanoid_object/<task>/`. Each task needs `scene.xml`, `task_info.json`, `0/trajectory_kinematic.npz`, and the assets referenced by the scene.

```bash
python experiments/paper.py omomo \
  --task move_largebox --dataset-dir /path/to/omomo_dataset \
  -- --output-dir runs/omomo_largebox
```
