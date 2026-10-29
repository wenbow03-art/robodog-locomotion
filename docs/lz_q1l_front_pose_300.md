# Side front posture continuation, 2026-10-29

User authorized the proposed rolling front-joint penalty and 300 additional updates.
Source: coordination model500 (sha256 e57308978f05ce3d6f91d61780cecb7a03bf9a216fdc36c3a065fac1dfde0b9b).

Only added reward: lateral_front_pose=-0.03. For actual front thigh and knee angles,
compute 1s sliding means; cost=sum((max(abs(mean_L-mean_R)-10deg,0)/10deg)^2).
Full 50-sample window required. Pure lateral only. Clear on reset/resample/command
change/inactive commands. Both front legs treated equally; rear and roll joints
have no direct term. All policy outputs may change indirectly through PPO.
URDF front pitch01/pitch02 axes are +Y with zero joint-origin rotation on both
sides; existing mirror map also uses +1 signs for pitch joints. No policy mirror.

Frozen actual model500 trace, last4s each: mean raw costs10.1795/9.5081; weighted
rates-.30539/-.28524, multiplied by .02 for environment per-step reward. Weight
chosen as moderate initial cost relative to lateral tracking scale3.0; no claim
of successful posture correction from this calibration.

Synthetic full-window, partial-reset, inactive, phase-alternation, exchange and
inference-tensor lifecycle checks passed. GPU32env100steps plus one discarded
PPO update passed: exact source weights and both Adam states, actor updated,
estimator unchanged, teacher absent, native actions. Main training starts fresh
from source, not the discarded check. Evidence: media/front_pose_review_20261029.

Train4096env100steps/update, lr1e-5, estimator frozen, no teacher/mirror,
left/right50/50, dt1ms decimation20, source optimizer restored, height-100,
lateral tracking3.0 and previous coordination rewards retained. Save0/50..300,
no quality early stopping. Stand/walk baseline checkpoint remains separate.

Evaluation after training should compare actual joint window gaps, both legs'
absolute posture and excursions, velocity, height, roll, slip, foot loads and
sustained suspension against source500. Equal means alone do not prove good gait;
both legs may converge to poor posture or compensate in other joints.

Run: /home/xiaohengli/Program/robo_competition/HIMLoco/legged_gym/logs2/flat_lz_q1l_front_pose/front_pose_300_20261029_164847_3445246
Status file is authoritative; launch pointer can be stale.
