# LZ-Q1L agent handoff

Read `deployment/HANDOFF.md` first, then `deployment/lz_q1l_straight_20261105/README.md`.
Paths in executable deployment tools are relative to the checkout; do not depend on the original server or chat history.

## Current task and state (2026-11-05)

The user selected `deployment/lz_q1l_straight_20261105/policy/model700_bands.pt` as the current straight-walking baseline and requested a cloneable handoff for hardware preparation. Model SHA-256: `109c87fb85abb565840990497a04aa003632cf164896b332073b9246545a5e30`.
The user will provide the robot address later. Do useful offline preparation first. The documented default `orangepi@192.168.8.10` is not a confirmed target.
No robot connection, calibration or motor control has been performed by this preparation workflow. The next step after receiving the address is read-only environment/state inspection.

## First executable check

Run `python3 deployment/check.py`; it needs only the standard library. For model inference use the environment instructions in HANDOFF and add `--model`. No Isaac Gym, GPU, historical training checkpoint or Git LFS is needed for this deployment check.

## Model semantics

- Load with `torch.jit.load`, not the historical PPO checkpoint loader.
- Infer at 50 Hz using six 45-value frames, newest first. Preserve real observation/action history. Routing state: int64 ones [1,14]; blend state: float zeros [1,2].
- The model includes stand/walk blending; do not add another stand mixer.
- Policy leg order RL, RR, FL, FR differs from SDK FL, FR, RL, RR. Map by name. Joint signs/zero offsets and IMU extrinsics are uncalibrated; null means unknown.
- The SDK already applies motor transmission/direction conversion. Do not apply it twice.
- This policy is for straight commands. Signed yaw rate, heading error and lateral offset are required; IMU alone does not provide lateral position. Do not silently substitute zero lateral offset.
- Offline/synthetic checks do not establish hardware readiness. Record remaining uncertainties and actual test scope.

## Working rules

Preserve source weights and historical results. Keep hardware captures, credentials, local calibration, caches and virtual environments out of Git. The provided tools only inspect and subscribe; preparing/cloning this repo is not authorization to stop the onboard controller or command motors.
Historical model selections and old running-process notes are archived at `deployment/history/agent_notes_20261029.md`; they are not current task state. The models under `models/` remain archived references, not the selected straight baseline.
For future training work, use the original V1-V8 config file and the shared `_2.py`/`_2_config.py` for the new lineage. Historical replay scripts require separate environment/source-path review; cloning this hardware handoff does not make every training launcher portable.
