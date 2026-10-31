# Third-party notices

## Do As I Do: object tracking metrics

`grasp/evaluation/tracking_error.py` is copied from Do As I Do
(https://github.com/malik-group/do-as-i-do, commit `824591b808c342b20079c3b4198a8c2bdf88c74e`),
`retargeting/retargeting/utils/tracking_error.py`. Only attribution comments have been added.

`grasp/pipeline/solve_ik.py` is adapted from `retargeting/retargeting/pipeline/solve_ik.py` of the same
commit (import paths and the processed-directory helper).

Source SHA-256: `9bb837fad227ca8692f5c6aed0650177e59021fe597abf1f0bde366fb372a1d4`.

MIT License

Copyright (c) 2026 malik-group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## SPIDER: configuration, data loading, MuJoCo Warp environment and tracking reward

Copied from SPIDER (https://github.com/facebookresearch/spider, commit
`44717007de41cbef7565dff7ff9f4453557a2d3d`) with import paths changed and the
fixes noted in the files:

| GRASP file | SPIDER file |
|---|---|
| `grasp/config.py` | `spider/config.py` (plus `load_config`) |
| `grasp/utils/io.py` | `spider/io.py` |
| `grasp/utils/interp.py` | `spider/interp.py` |
| `grasp/utils/math.py` | `spider/math.py` |
| `grasp/sim/mjwp.py` | `spider/simulators/mjwp.py` |
| `grasp/sim/rollout.py` | `make_rollout_fn` from `spider/optimizers/sampling.py` |
| `grasp/utils/mujoco_utils.py` | `spider/mujoco_utils.py` |
| `grasp/pipeline/oakink.py` | `spider/process_datasets/oakink.py` (trimesh instead of pymeshlab, per-task object assets) |
| `grasp/pipeline/decompose.py` | `spider/preprocess/decompose.py` |
| `grasp/pipeline/decompose_fast.py`, `grasp/pipeline/_decompose_common.py` | `spider/preprocess/decompose_fast.py`, `spider/preprocess/_decompose_common.py` |
| `grasp/pipeline/generate_xml.py` | `spider/preprocess/generate_xml.py` (optional object connections) |
| `grasp/pipeline/ik_fast.py` | `spider/preprocess/ik_fast.py` (object quaternions aligned and renormalized in the moving average; no viser viewer) |
| `grasp/pipeline/gmr.py` | `spider/process_datasets/gmr.py` (robot model directory as an argument) |
| `grasp/assets/mano/empty_scene.xml` | `spider/assets/mano/empty_scene.xml` |
| `config/default.yaml`, `config/override/{oakink,humanoid,humanoid_object}.yaml` | `examples/config/` |

These files remain under SPIDER's license:

CC-by-NC License
Copyright (c) Meta Platforms, Inc. and affiliates.

You are free to share (copy and share the work in any way) and adapt (change or
build upon the work), but you must give credit to the original creator and not
use the work for commercial purposes. You cannot use the work for anything that is
primarily intended for commercial advantage or monetary compensation.

For full details, visit: https://creativecommons.org/licenses/by-nc/4.0/

## TopoRetarget: Wuji hand description

`grasp/pipeline/graft_wuji.py` reads the Wuji Hand 2 description (`robots/wujihand2_{right,left}.json` and the MJCF
they reference) from TopoRetarget (https://github.com/wuji-technology/toporetarget, commit
`039c54022016a882ef8d3a6d34b4578aee7cb51f`). The original configurations, scenes and referenced hand model are
bundled under `grasp/assets/robots/wuji/`, preserving their relative paths. The hand description's license is
included at `grasp/assets/robots/wuji/assets/wuji-description/LICENSE`.

MIT License, Copyright (c) 2026 Wuji Technology.

## Bundled robot models

`grasp/assets/robots/{sharpa,xhand,unitree_g1,booster_t1}` contains the original robot assets used by the evaluation
datasets, copied without changing model geometry or physical parameters. Sharpa comes from the OakInk2 evaluation
assets; XHand, G1 and T1 come from SPIDER's example dataset robot assets. Existing model license and attribution
files are retained, including `grasp/assets/robots/unitree_g1/LICENSE` (BSD 3-Clause, Unitree Robotics).

## HOT3D: MANO hand model

`grasp/pipeline/hot3d_mano.py` is the MANO hand model file of the HOT3D toolkit
(https://github.com/facebookresearch/hot3d), with a source line added.

Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0).
