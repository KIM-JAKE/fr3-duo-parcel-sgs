# FR3 Duo parcel SGS PPO

Portable snapshot of the native FR3 Duo + two Tesollo DG-5F-M parcel experiment, including exact-computation physics optimizations, data integrity checks, a 25% NG free-success sampling budget and durable training launcher.

**Dataset:** [jaeikkim/fr3-duo-parcel-resets-200k](https://huggingface.co/datasets/jaeikkim/fr3-duo-parcel-resets-200k) — public, automatically gated.

## New server

Linux, Python 3.12, NVIDIA driver supporting CUDA 13.0, `git`, `tmux`, `uv`. For headless MuJoCo install the distribution's OSMesa/OpenGL runtime (`libosmesa6`, `libgl1` on Ubuntu). RTX/RT cores are not required for the current privileged teacher. The tested large configuration uses **four B200 180-GiB GPUs**, approximately 155–161 GiB resident per GPU. Other devices require their own capacity benchmark.

```bash
git clone https://github.com/KIM-JAKE/fr3-duo-parcel-sgs.git
cd fr3-duo-parcel-sgs
bash scripts/install.sh
source .local_isaaclab3/venv/bin/activate
hf auth login  # enter a read token interactively, accept dataset gate first
python scripts/prepare_data.py --data /your/data/parcel-resets --revision 57561a95da538590bca3dc83d63c108f67eef43f
python scripts/preflight.py
# FIRST run a small real four-GPU update before committing to the large configuration.
tmux new-session -d -s parcel_smoke \
  'cd "'"$PWD"'" && .local_isaaclab3/venv/bin/python scripts/train.py --gpus 0,1,2,3 --mailer-envs 4 --iterations 1 --horizon 2 --episode-steps 4 --diagnostic --output runs/smoke'
tmux attach -t parcel_smoke
```

Check `runs/smoke/supervisor.json`, all four rank statuses and `latest_checkpoint.json` before proceeding. Preparation verifies the downloaded SHA256 manifest and writes server-local asset/shard paths without altering the downloaded release.

### Default: 32k/GPU, shorter PPO rollout

The default retains32,768 environments/GPU and uses horizon8, giving1,048,576 transitions per four-GPU update. The physical simulation and maximum environment population stay unchanged; shorter rollouts change update cadence, not physical env/s. Measure end-to-end env/s including collection, optimizer and scene rebuilds. Smaller configurations are diagnostics or explicit memory-constrained alternatives, not an automatic substitute for the requested population.

```bash
tmux new-session -d -s parcel_train \
  'cd "'"$PWD"'" && .local_isaaclab3/venv/bin/python scripts/train.py --gpus 0,1,2,3 --output runs/sgs'
```

### Large capacity setting

```bash
tmux new-session -d -s parcel_train \
  'cd "'"$PWD"'" && .local_isaaclab3/venv/bin/python scripts/train.py --gpus 0,1,2,3 --mailer-envs 4096 --horizon 32 --output runs/sgs32k'
```

This gives 32,768 environments/GPU = 16,384 box + 4×4,096 mailer environments, total 131,072. One shared PPO actor per GPU, synchronized across all GPUs. **This is the tested maximum capacity setting, not a guarantee of optimal throughput.** Original end-to-end production updates took **4,419–4,456 seconds per 32-step rollout/update**. A policy step contains ten .002s native physics calls; large contact-rich deformable batches dominate runtime. `evidence/` contains measurements and exact scope. Use `--mailer-envs N` to scale both kinds proportionally (8N total/GPU) while keeping physical fidelity unchanged.

### Resume and monitoring

Rerun the same launcher with the same output directory under tmux after verifying old owned processes exited. It verifies SHA256 for all rank checkpoint files and restores optimizer, SGS history and worker RNGs. Before a first checkpoint, restart begins from recorded initialization. Episodes restart from the pinned reset pool; exact mid-trajectory continuation is not promised. Missing/stale statuses fail closed. Logs append; no checkpoints are deleted.

- `runs/sgs/launch.json`: exact command, PID, GPUs, reset budget, resume source.
- `runs/sgs/supervisor.json`: current health and learner statuses.
- `runs/sgs/train.log`, `workers_rank*/worker_*.log`: logs.
- `runs/sgs/latest_checkpoint.json`: SHA-verified checkpoint set.

The complete launcher runs within tmux and survives client/SSH disconnect. Avoid running training directly in a Codex/SSH foreground tool session.

## Reset / learning protocol

200k selected reset states; 50k each R/NO/G/NG, half boxes and half deformable mailers. Exact geometry/material per native simulator. Geometry-batched synchronous episodes, absorbing terminated worlds and fixed 50/50 kind strata are explicit adaptations of SGS, not an identical paper benchmark. Privileged 256D teacher observation, 54 joint actions, RSL-RL PPO 512/256/128, horizon8, gamma.99, lambda.95, four epochs/four minibatches. No RGB training or distillation yet.

Goal: whole parcel footprint on output support, bottom height within15mm, label-up score>=.8 for three consecutive policy frames, plus contents enclosure on mailer success candidates. Reward terminal+100/failure−1, SGS action/action-rate/velocity regularizers, label/goal-distance potential shaping.

### NG free-success budget

Raw original NG no-action successes were 72.656% box /79.956% mailer over a **0.4s** probe; zero samples were already successful at reset. The new sampler preserves physical states but enforces **25% of NG probability mass for those free-success states and75% for the rest, separately per kind**, after every SGS weight calculation. This is a marginal sampling target; homogeneous batches fluctuate. It does not claim raw-pool composition changed or a25% baseline over16s episodes. `scripts/preflight.py` validates fresh and updated-history probabilities.

The original checkpoint after two updates had NG mailer6804/6806 completed successes, NO0/14417, R0/74, G0/2595. Box had only one completed NO episode (failed); the other box families had no completed trials. This is insufficient evidence of learned flipping. Baseline free success and learned manipulation success must be reported separately. The biased original training run was stopped and checkpoints preserved locally; fresh runs are the default here.

## Physics optimization and provenance

No reduced mesh, contact physics, timestep or solver iterations. Sorted native contact gathering retains force/Hessian accumulation order. Specialized face kernels retain24 Frank-Wolfe iterations/16 line-search iterations with exact fixed-point skipping. Native self-contact CSR filters are reused only after all world topologies match. Bitwise frozen-state parity proofs and template hashes are included under `fr3_duo_sgs/runtime/*equivalence*`.

Matched32k/GPU held-state probes measured mailer step reductions8–23% and solver initialization5–6.7× faster; these are not whole-PPO speedup claims. Sources remain pinned to IsaacLab commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`, Newton1.6.1, Warp1.17.0, RSL-RL5.5.1, Torch2.11.0+cu130. Critical installed Newton file hashes fail closed in preflight. The fresh-server installation and full4GPU rollout must still be checked on the destination hardware.

Upstream source hashes, installed versions and deliberate portability/sampler changes are recorded. The original monolithic machine-specific workflow is not exported as a launch path. Original assets originate from Franka and Tesollo; their licenses and pins are included. See `fr3_duo_sgs/NEWTON_OPTIMIZATION_NOTICES.md` for derived kernel notices. Hardware validation, fully successful learned parcel handling, and globally optimal runtime are not claimed.

Dataset release SHA: `57561a95da538590bca3dc83d63c108f67eef43f`. Prefer this immutable revision over `main`.

Mixed five-simulator measurements on B200 (same four held mailer sources,8warm+24measured .002s physics steps):4,096 total environments/GPU used23.42GiB and maximum mailer-shard step1.55s;8,192 used41.88GiB and maximum shard step3.00s. Original32,768 used152.09GiB and maximum shard step10.87s. These are physical-step latency measurements, not yet end-to-end PPO update timings for the new presets.

## Policy timing: OmniReset main setting (2026-11-16 revision)

The official OmniReset state-RL configuration uses physics dt1/120s, decimation12 and episode length16s: **policy10Hz, maximum160 policy steps**. Source: https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py#L635-L639

This task adopts **10Hz/160 steps**, while retaining its verified Newton dt.002s, two internal substeps and VBD40. Each policy action advances50 native physics calls. Joint velocity actions are converted to a reference interpolated at the native physics rate, preserving per-second joint speed instead of retaining a50Hz-sized displacement. Collision/failure/success checks remain at50Hz, including the existing three-audit success hold; success is latched once if reached within a100ms policy action. Terminal/failure/potential rewards and gamma.99 remain per policy decision; existing regularizer coefficients remain per decision rather than being silently rescaled. Controller timing and interpolation change trajectories; frozen-kernel proofs concern the unchanged low-level kernels, not trajectory identity across policy rates.

Changing50Hz to10Hz does **not** reduce fine physics computation per simulated second. With a fixed number of PPO rollout steps it increases simulated duration and physics work per update by5x. The16s episode cap is a timeout, not the desired parcel completion time. NG free labels still refer to their original0.4s replay; no25% baseline over16s is claimed.

`control_timing.py` is the timing authority. Checkpoints include timing metadata and refuse incompatible50Hz/10Hz resumes. Use a new output directory for this revision. `evidence/timing_contract.json` exercises the actual control loop with synthetic dynamics:50 physics calls/action,8000 calls/160 steps,16s timeout, native-rate action scaling and one-time terminal event latching. A physical multi-GPU smoke is recorded separately; synthetic tests are not manipulation evidence.

Portable10Hz native smoke and resume passed on4 B200 GPUs:128 total diagnostic environments,20 native workers, actual PPO optimizer16→32 steps/rank, actor weights changed on all4 ranks, SHA-verified checkpoints, global SGS history equal and5 worker RNG states restored/rank. See `evidence/portable_10hz_smoke_resume.json`. Diagnostic timeout4 steps is intentionally short; the160-step/16s control accounting is covered separately. The prior32k/GPU capacity proof uses the previous50Hz controller and is not relabeled as a full32k/GPU10Hz validation.
