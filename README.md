# GRASP

### Gradient-Guided Sampling for Contact-Rich Physics-Based Motion Retargeting

**Yang Liu\*, Tong Wang\*, Yiming Bao, Heng Yang**  
Harvard University · \*Equal contribution

[Project Page](https://computationalrobotics.seas.harvard.edu/GRASP) · [Data Preparation](docs/data.md) · [Paper Configurations](experiments/paper.py)

GRASP turns human motion references into robot controls under contact dynamics. It recovers gradients from archived simulator rollouts to guide sampling, without requiring a differentiable simulator.

- **GRASP-TO** fits local quadratic cost models to archived samples and uses their gradients in Metropolis–Hastings-corrected proposals within tempered sequential Monte Carlo.
- **GRASP-MPC** applies this optimizer in receding windows, carrying plans and samples forward to track long demonstrations.

The retargeting pipeline builds a robot–object scene, solves inverse kinematics, and refines the reference through physics simulation. The paper evaluates dexterous hands and humanoids with a shared state-tracking objective and no task-specific contact rewards.

## Installation

Use Python 3.12 and an NVIDIA GPU with a CUDA-capable driver for MuJoCo Warp. From a machine with [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/ComputationalRobotics/GRASP_code.git
cd GRASP_code
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[pipeline,benchmarks,dev]"
```

Dependencies are specified in [`pyproject.toml`](pyproject.toml). `pipeline` installs data preparation and IK dependencies; `benchmarks` adds plotting; `dev` adds test tools.

Run the commands below from the repository root. For GRASP-MPC, keep JAX on CPU while MuJoCo Warp runs simulation on GPU:

```bash
export JAX_PLATFORMS=cpu
```

## Run GRASP-MPC

First [prepare a dataset](docs/data.md), including its object meshes and kinematic references. Robot models are included in [`grasp/assets/robots`](grasp/assets/robots); demonstration data is supplied separately.

For a prepared Sharpa task:

```bash
python experiments/paper.py oakink2 \
  --task oakbi_443_take_outside \
  --robot sharpa_bimanual \
  --dataset-dir data/oakink2 \
  -- --output-dir runs/sharpa_443
```

`experiments/paper.py` selects the dataset's optimizer settings and derives the horizon from the reference. Task `443` is a short, 0.4-second example at the paper's 50 Hz reference rate. It uses the same 2,000-particle hand configuration as the longer demonstrations.

| Dataset argument | Robot argument | Embodiment |
|---|---|---|
| `oakink2` | `sharpa_bimanual`, `xhand`, `wuji_topo` | `bimanual` |
| `hot3d` | `sharpa` | `bimanual`, `right`, `left` |
| `amass`, `lafan` | `unitree_g1` | `humanoid` |
| `omomo` | `unitree_g1` | `humanoid_object` |

Select the embodiment with `--embodiment` when it differs from the dataset default. The task's scene and reference must already be prepared for the selected robot and embodiment.

Useful options:

- `--print`: show the resolved MPC command without running it.
- `--horizon-rounding up`: use upward knot rounding for task sets prepared with that convention.
- Arguments after `--` are forwarded to `grasp.mpc.run`, for example `--seed 1` or `--override nconmax_per_env=512`.

Hand runs use 2,000 particles, 15-knot windows, a commit count of 2, and a 0.2 s knot spacing (0.1 s for references shorter than one standard knot). Humanoid runs use 1,000 particles, 8-knot windows, a commit count of 1, and a 0.1 s knot spacing. See [`experiments/paper.py`](experiments/paper.py) for the complete settings and horizon conventions.

### Outputs

Each run writes matching `.json` and `.npz` files under `--output-dir`:

- **JSON:** arguments, tracking metrics, costs, runtime, and per-window optimization records.
- **NPZ:** executed positions and velocities, committed controls, replay costs, and per-frame object errors.

For hand tasks, object position errors are in **meters** and rotation errors in **radians**. Success requires mean position error below 0.1 m and mean rotation error below 0.5 rad. The runner also replays the committed controls and reports the cost difference from online execution.

## Trajectory Optimization Benchmarks

The acrobot and planar pusher-slider benchmarks compare GRASP-TO with gradient-based tempered SMC, random-walk Metropolis tempered SMC, and independent MPPI instances per particle.

```bash
python -m benchmarks.run --task acrobot --method grasp --seeds 0-6
python -m benchmarks.run --task pusher --method grasp --seeds 0-10
```

Choose `--method tsmc_grad`, `rwm`, or `mppi` for the baselines. Results are written to `outputs/benchmarks` by default. [`benchmarks/run.py`](benchmarks/run.py) contains the task settings and budgets.

## Repository Layout

| Path | Contents |
|---|---|
| [`grasp/optim`](grasp/optim) | Tempered SMC, surrogate fitting, and AR(1) priors |
| [`grasp/mpc`](grasp/mpc) | Receding-horizon controller and run entry point |
| [`grasp/sim`](grasp/sim) | Batched MuJoCo Warp simulation |
| [`grasp/pipeline`](grasp/pipeline) | Dataset conversion, object decomposition, scene generation, and IK |
| [`grasp/evaluation`](grasp/evaluation) | Object tracking metrics |
| [`grasp/assets/robots`](grasp/assets/robots) | Sharpa, XHand, Wuji, Unitree G1, and Booster T1 models |
| [`config`](config) | Simulator and tracking-cost configuration |
| [`experiments/paper.py`](experiments/paper.py) | Dataset-specific paper configurations |
| [`benchmarks`](benchmarks) | Acrobot and pusher-slider experiments |
| [`tests`](tests) | Optimizer, surrogate, and controller tests |

Run the tests with `python -m pytest -q tests`.

## License

GRASP is released under the [MIT License](LICENSE). Third-party code and models keep their own licenses, listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Acknowledgments

This repository incorporates code and assets from [SPIDER](https://github.com/facebookresearch/spider), [Do As I Do](https://github.com/malik-group/do-as-i-do), and [TopoRetarget](https://github.com/wuji-technology/toporetarget). Physics simulation uses [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), and IK uses [mink](https://github.com/kevinzakka/mink).

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the bundled model licenses for attribution and component-specific terms.
