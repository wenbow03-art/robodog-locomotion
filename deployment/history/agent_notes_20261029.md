# LZ-Q1L simulation and recording defaults

## Soft slip300 authorized and launched (2026-10-29)

- User reviewed softslip change and authorized300additionalupdates from final
  frontposemodel300. Reviewed reward/config hashes unchanged;warmreuse passed
  frozenGPUpreflight. Launch`tools/launch_lz_q1l_soft_slip.py`,trainer/wrapper
  samebasename. Noqualityearlystop;save0/50/100/150/200/250/300.
- Run`soft_slip_300_20261029_181955_3497700` underlogs2/flat_lz_q1l_soft_slip;
  GPU0,tmux`lz_q1l_soft_slip_300`. Pointer`media/soft_slip_20261029/latest_training.json`.
  Read run/status.json foractualprogress. Do notuse prior stagecompleted status.
- Newreward-.05 only;1ms/20,4096env,100steps,lr1e-5,sourceweights+bothAdam,
  estimatorfrozen,noteacher/nomirror,left/right50/50. Trainingauthorization
  supersedes oldreview-only notesbelow;no otherstage authorized.

## Soft slip implemented, awaiting user review (2026-10-29)

- User authorized code changes for review ONLY; no training launched.
- Task`lz_q1l_flat_soft_slip`,env`LZQ1LSoftSlipRobot`,cfg`LZQ1LSoftSlipCfg/PPO`
  appended in shared_2 files. Source finalfrontpose300,sha66ff4a30...bf6bd88c0.
- Onlynewreward lateral_soft_slip=-.05:allfourfeet,pureside;forceweight
  smoothstep(clamp(Fz/5,0,1));z=max(bottomXYspeed-.03,0)/.10;
  loss=z² if z<=1 else2z-1. Sumfeet,standarddt.02 scaling. No contact-duration
  gating fornewcost. Noairforce =>zerocost. Bottomsphere rigidbody velocity.
- Otherrewards/config preserved inclfrontpose-.03;no rearposeconstraint.
  Freshsourcecfg usesexistingHIMSideCoordinationRunner(iter300restore),frozen
  estimator,bothAdamrestore,noteacher/nomirror. No launcherauthorized.
- Synthetic+fixedtracecalibration+GPU32env100steps FROZENpolicy passed;
  zeroPPOupdates,model/optimizerunchanged. Evidence`media/soft_slip_review_20261029`.
  Meanweightedrates-.06516/-.10589,totalperstep-.001303/-.002118.
  Frontlegs receive mostnewcost;do notdescribe as rightrear-onlyrepair.
- Review`docs/lz_q1l_soft_slip_review.md`;awaituserreview/traininginstruction.

## Front pose300 active (2026-10-29)

- User approved rolling front thigh/knee posture penalty and300 additionalupdates
  from coordination model500. Task`lz_q1l_flat_front_pose`,new env/config appended
  in shared_2 files; tracker`utils/lz_q1l_front_pose.py`.
- Only new reward scale-.03:1s averageactual angle L/R difference,10degdeadband,
  normalized10deg squared sum;fullwindow;clear onreset/resample/change/inactive.
  Both frontlegs,onlypitch01/pitch02,pure lateral;no actioncopy/mirror/teacher.
- Exact model+bothAdam restore,estimatorfrozen,1ms/20,left/right50/50,4096envs,
 100steps,300additionalupdates,save50,noqualitystop. Allotherrewards unchanged.
- CPU/calibration/GPU32env100steps+oneDISCARDEDupdatepassed;warmenvreuse.
  `docs/lz_q1l_front_pose_300.md`;`media/front_pose_review_20261029/` evidence.
- Launcher`tools/launch_lz_q1l_front_pose.py`;pointer
  `media/front_pose_20261029/latest_training.json`;tmux`lz_q1l_front_pose_300`.
  Read run/status.json for actual progress;never reuse stale pointer counters.

## Side coordination500 active (2026-10-29)

- Launch verified:PID3332497 onGPU0, firstupdatecompleted (~11.2s/update),
  actualGPU8618MiB, exactsource/optimizer auditpassed;coordination metrics present.

- User approved reviewed coordination changes and500additionalupdates. Launcher
  `tools/launch_lz_q1l_side_coordination.py`;trainer/run wrappers similarly named.
- Start final1msheightmodel300, exact model+bothAdam restore;1ms/decimation20,
  left/right50/50,no mirror/no teacher,frozenestimator. Reviewed3rewards unchanged.
  Runtime maxiterations override500 only; reviewed defaultconfig remains300.
- Every50completedupdates saved atomically with exact stageiteration;model0 plus
 50..500. No qualityearlystop/automaticdeployment/nextstage. Stand/walk unchanged.
- Warm reuse of passedCPU/GPUpreflight with exact reviewedcode/config verification;
  newtrainer audits source/optimizer onlaunch,estimator everyupdate andbaseline50.
- Pointer `media/side_coordination_20261029/latest_training.json` ->run/status.json;
  submittedGPU0,tmux`lz_q1l_side_coordination_500`,run
  `side_coordination_500_20261029_130610_3332474`. Read status foractualprogress.
- Perdirection/perfoot coordinationmetrics inTensorBoard andstatus.json;no further
  teacher or estimator training. Prior reviewed snapshots preserved immutable.

## Side coordination changes implemented, awaiting review (2026-10-29)

- User authorized implementing proposed3changes forreview, NOT formaltraining.
  New task`lz_q1l_flat_side_coordination`;env`LZQ1LSideCoordinationRobot` in_2.py,
  cfg`LZQ1LSideCoordinationCfg/PPO` in_2_config.py;no4ms tasks changed.
- Source final1msheightmodel300 SHAb8984209bcb1711addf2bfda6739c7b7e0400d280e42d6638f66c8d26e56ec11.
  DedicatedHIMSideCoordinationRunner restores exactweights/Adam,freezesestimator,
  no teacher/nativebothdirections;current stagecounter0. No training launched.
- Side-only:1srolling perfoot force share10%floor,normalized squared deficits
  summed,weight-1;requires fullwindow andtotal>=10N. Continuous ineffective
  support>0.30s normalizedsquare cost,perfootcap4,weight-.5. Stablecontact>=5N
  for2control samples(40ms counted). Reset/windowclear oncmdchange/reset/inactive.
- Override side feet_air_time (same+.3scale) with validlanding reward:previous
  support->air->stablecontact;>=1cmcollisiongap;D<=.06zero,.06-.15linearramp,
  .15-.30capped1,>.30zero. Firstlanding not rewarded;onetimeevent,no tapreset.
  Non-side old feet_air_time preserved. All4feet same rules,allowalternation.
- Keep1ms/20,height-100,target.30,tracking3,LR1e-5,left/right50/50,no mirror.
  Added perfoot force/airtime/contact/slip logging;no slip reward added.
- CPU+native-trace calibration+GPU32env100steps/oneDISCARDEDupdate passed. Source
  and optimizer exact,estimatorunchanged,actorupdated,lifecycle tested. Initial
  tracker inference-tensor rebinding fixed; testcontextcorrected; failures archived.
- Review`docs/lz_q1l_side_coordination_review.md`;evidence/manifests/provenance in
  `media/side_coordination_review_20261029/`. Native trace meanweightedrate costs
  leftload-.191/air-.0266/landing+.0148,right-.1817/-.0025/+.0127;not learningresults.
- Await userreview/trainingcommand. No launcher/formalrun started. On authorization
  use audited trainer saving exact50-update checkpoints and immutablemetadata;
  default cfg300 is not trainingauthorization. Do not load oldV8 as start.

## User direction:1ms native only; coordination proposal (2026-10-29)

- Latest user: no further4ms training; focus1ms, native left/right (mirror no
  longer needed). Both prior runs alreadycompleted. No new training authorized
  by this question; user requests ideas for coordination and sustained airtime.
- New working reference: final1msheight model300, native both sides. Keep old
  checkpoints and baseline stand/walk intact; no automatic deployment.
- Last4s nativeleft RH force8.90N/share5.8%,no-contact81.5%,longest.54s at1N
  (.56s at5N). Nativeright LF8.57N/share5.6%,no-contact65%,longest.36s.
- Current feet_air_time reward positiveweight.3 computes(sum((airtime-.3)*
  first_contact)); longer single swing earns larger landing term. Possible
  conflict with shorter sustainedairtime, not proved sole cause.
- PROPOSAL ONLY: side-only perfoot1s rolling vertical-load floor10% total;
  perfoot continuous-airtime cost after tentative.30s; bound/rework old
  positiveairtime reward so no extra incentive beyondtarget; stable contact
  e.g5N for40ms resetsclock to avoid flicker. Allfourfeet, bothdirections,
  allowphasealternation. Calibrate weights on current traces beforetraining.
- Keep dt1ms/decimation20,height-100,tracking3,frozenestimator,no teacher/mirror.
  Evaluate native speed,height,roll,loadshares,airtime durations and contactslip
  (slip must be newly logged; existing traces lack footvelocity),not forced25%
  instantforce or allfeet100%contact. No reward changes implemented this turn.

## 1ms model300 native-left replay (2026-10-29, latest)

- User requested no mirror and narrowed to1ms only. Replayed final1msmodel300,
  pure native bothdirections,own2s stand warmup,left+.30/right-.30 each5s,
  continuous,no stage resets. No training/config/deployment changes.
- `media/side_height1ms300_native_20261029/latest_result.json`;video
  http://localhost:8769/side_height1ms300_native_1ms_run_20261029_124122_zunI/
- Last4s LEFTnative:vy+.27784m/s,height28.08cm,rollabs1.159deg,forward+.1180m/s.
  ForcesLH/RH/LF/RF84.26/8.90/34.93/26.01N;contact91.5/18.5/67/88%.
  Nativeleft DOES have tracking ability now; abnormal lowload isRH, notRF.
  Previous mirroredleft:vy+.27856,height28.14,roll4.897,forward+.0972;lowloadRF.
- RIGHTnative after nativeleft:vy-.27121,height28.18,roll5.552,forward+.0925;
  forces51.58/26.87/8.57/67.00N;LF still lowload. Different preceding history.
- No resets/falls; exact source/config/checkpoint hashes,native-action equality,
  full250frames/10sdecode,HTTP200+preview passed. Single nominal trial only;
  cannot claim allcommandsrobust or independentnativeleft/right equality.

## Front-leg coordination diagnosis (2026-10-29)

- User observes right-side steppingLF abnormal,mirroredleftRF abnormal. Existing
  final300 trace confirms, no new simulation/training/reward changes.
- Last4s RIGHT:4ms LF10.96N/contact21.5% vsRF41.64N/95.5%;LFmean sphere
  gap67.79mm,max207.81mm.1msLF8.92N/37.5% vsRF66.83N/77.5%;LFgapmean6.12mm,
  max25.89mm. Distinct patterns:4ms excessive lift/swing;1ms lowload foldedleg.
-1ms LF pitch01 targetmean-57.14deg vsRF+.30deg,pitch02+24.44 vs-11.45deg.
  These are commanded joint targets,not measured joint positions. Joint-order
  source oldstanding manifest with SAME URDFhash3c4a5128...7d59dda.
- Leftmirrored case maps problem toRF asexpected. No evidence this is a mirror
  implementation bug. Aggregate velocity/height success doesnotensure leg load
  participation; current rewards lack explicit perfoot rolling loadfloor.
- `media/side_height300_compare_20261029/front_leg_diagnosis.json` stores contact,
  forces,gaps,targetstats. Single nominal traces;no causal reward ablation.
- Proposed only (not implemented): side-only perfoot minimum share of1s averaged
  vertical support (initial10%softfloor),allfourfeet to avoid moving lowload issue
  elsewhere; retain speed/height,allow alternating support,validate beforetraining.

## Final height300 both timesteps replayed (2026-10-29)

- BOTH4ms and1ms training completed300/300 exit0,all50 checkpoints. No jobs remain
  for these two runs; latest_training pointers updatedcompleted. No new training.
- Replayed eachFINALmodel300 at its own traineddt. Same protocol: own2s native
  stand warmup,left+.30/right-.30 each5s,continuous,left MIRROREDright,rightnative,
  no stage resets,noise/randomization off. This does NOT test native-left learning.
- 4msvideo http://localhost:8769/side_height4ms300_4ms_run_20261029_122930_eP9z/
  Last4s L/R speeds+.3244/-.3371m/s,height25.27/25.37cm,rollabs3.10/3.53deg,
  forwarddrift+.0296/+.0358m/s. ForcesLH/RH/LF/RF left57.52/37.44/42.28/10.66N,
  right40.17/57.96/10.96/41.64N.
- 1msvideo http://localhost:8769/side_height1ms300_1ms_run_20261029_122934_RaVr/
  Last4s L/R speeds+.2786/-.2699m/s,height28.14/28.18cm,rollabs4.90/5.57deg,
  forwarddrift+.0972/+.0923m/s. Forcesleft27.35/50.09/67.69/8.28N,
  right51.32/26.63/8.92/66.83N. Heightcloser30cm,buttilt/drift/loadimbalance remain.
- Neither replay fell/reset. Strict checkpoints/config hashes;4ms only prior
  verifiedappend-only1ms declarations exception;1ms all exactsourcehashes.
  Both250frames/10s/full decode+HTTP200+preview verified. No deployment.
- Pointers `media/side_height4ms300_20261029/latest_result.json`,
  `media/side_height1ms300_20261029/latest_result.json`;comparison
  `media/side_height300_compare_20261029/comparison.json`. Single trials,not
  robustness proof;different policies AND dynamics,not pure timestep ablation.

## Height4ms model200 replay (2026-10-29)

- User requested4ms200. Used HEIGHT-100 stage model200, not prior speed model200.
  Replay inherits latest userleftmirror request: left mirroredright,right native;
  training remains unmirrored.4ms/decimation5,own2s stand warmup,left+.30/right-.30
  each5s,continuous,no stage resets,noise/randomization off.RecordingGPU2.
- `media/side_height4ms200_20261029/latest_result.json`;video
  http://localhost:8769/side_height4ms200_4ms_run_20261029_004549_NEE9/
- Last4s:side+.3414/-.3212m/s,height24.92/24.90cm,rollabs3.58/4.79deg.
  Contact LH/RH/LF/RF left83/87.5/95.5/21.5%;forces52.79/40.89/43.64/12.50N.
  Rightforces40.38/52.29/14.34/42.80N. No falls/resets. Height improved vs
  previous mirrored speed200video17.49/17.65cm,still below30cm;load imbalance remains.
- Exact training config and checkpoint verified. Two sourcefiles __init__.py and
  _2_config.py have ONLY appended1ms classes/registration since training; original
  archived prefixes verified bytewise+SHA,appended AST scoped and suffixes/current
  hashes pinned in selected.json and replaymetadata. Allother code hashes exact.
- Full250frames/10sdecode andHTTP200 verified. Neither training job stopped.

## Parallel direct1ms height300 active (2026-10-29)

- Actually runningGPU1,PID2973768,tmux`lz_q1l_native_side_height1ms_300`,
  run`native_side_height1ms_300_20261029_002802_2973756`;firstupdate verified,
  ~52.8s firstupdate (variable sharedGPU load).4ms continuesGPU3.

- User requests keep height settings and train1ms simultaneously. Do NOT stop
  running4ms height job. New task `lz_q1l_flat_native_side_height1ms`,300/save50.
- Same source side_speed model250 SHA d6457eac165260a133a9fbeaff04015198621f915dbc945c33acf3a81f4478e4,
  full model/optimizer restore. Height-100,target.30,tracking_lateral3.0,
  no teacher,no mirror,frozen estimator,left/right50/50 unchanged.
- Only changes vs actual4ms run:dt.004->.001,decimation5->20,delay_steps
  [0,1,2,4]->[0,4,8,16] to preserve0/4/8/16ms physical delays. Policy still50Hz.
- New cfg appended `_2_config.py`; launcher
  `tools/launch_lz_q1l_native_side_height1ms.py` excludes4ms jobGPU.
  Status `media/native_side_height1ms_20261029/latest_training.json`->run/status.json.
- Preflight01 checks saved4ms config delta,source,optimizer,finite update,frozen
  estimator,balanced native actions. Details`docs/lz_q1l_native_side_height1ms.md`.

## Height recovery continuation300 active (2026-10-29)

- Actually running onGPU3,PID2967087,tmux`lz_q1l_native_side_height_300`,
  run`native_side_height_300_20261029_002230_2967074`;first update and GPU verified.

- User requested stronger height cost and confirmed300 additional updates.
- Transition old side_speed job after checkpoint250 is saved; keep it as source,
  terminate only PID2935842 after command-line identity verification. This is
  user-requested reward change, not a quality automatic early stop.
- Old job stopped at exactly250; source SHA d6457eac165260a133a9fbeaff04015198621f915dbc945c33acf3a81f4478e4.
- New task `lz_q1l_flat_native_side_height`, newcfg in `_2_config.py`, sole env
  delta base_height-10->-100,target.30m. Restore model250 and both Adam states;
  estimator stays frozen, no teacher or mirror. LR1e-5,side tracking3.0,
  4ms/decimation5,left/right50/50 unchanged. New300/save50,independent stage count.
- Source selection `media/native_side_height_20261029/selected.json`; actual
  launch/progress `latest_training.json` in same folder -> run/status.json.
  Launcher `tools/launch_lz_q1l_native_side_height.py` requires preflight01.
- Details `docs/lz_q1l_native_side_height.md`. No deployment/nextstage/quality stop.

## Speed model200 left-mirror replay override (2026-10-29)

- User explicitly requested left using mirrored right during simulation.
  Replay-only: same teacher-free speed model200,4ms/decimation5,left/right.30 each5s,
  own2s native stand warmup,continuous switch,no stage resets. Training remains
  native50/50 without teacher; global task factory unchanged.
- `media/side_speed200_mirror_20261029/latest_result.json`,video
  http://localhost:8769/side_speed200_mirror_4ms_run_20261029_001445_LNfN/
- Last4s left+.32125/right-.30543m/s;height17.49/17.65cm,rollabs1.31/1.74deg.
  No falls/resets. Mirroring improves left speed vs native+.10234, but severe
  crouch persists. Right preceding physical history differs from native replay.
- Strict hashes/config,reflected-left/raw-right action equality,250frames/10s
  and full decode verified. RecordingGPU3; no training changes/stops.

## Teacher-free speed model200 replay (2026-10-29)

- User requested model200 during ongoing300 training; training NOT stopped.
  At replay completion it was211/300. Read run/status.json for current progress.
- `media/native_side_speed200_20261029/latest_result.json`, task native_side_speed,
  no teacher,tracking_lateral3.0. Continuous left+.30/right-.30 each5s, own2s
  standing warmup,4ms/decimation5,NO MIRROR,no stage resets,noise/randomization off.
- Last4s side speeds+.1023/-.3415m/s,height18.66/17.58cm,rollabs1.86/1.71deg.
  No falls/resets but severe crouching; not a posture success. Forces LH/RH/LF/RF:
  left56.55/30.47/46.94/16.71N,right36.43/56.18/21.82/33.80N.
- Strict training code/config/checkpoint hashes and native-action equality passed;
  250frames/10s/full decode and HTTP200 verified. GPU3 recording,trainingGPU2.
- http://localhost:8769/native_side_speed200_4ms_run_20261029_000758_vw0H/
- Single nominal continuous replay; different direction histories,not robustness.

## Teacher-free side speed candidate (2026-10-28, latest change)

- Actually launched GPU2,PID2935842,tmux`lz_q1l_native_side_speed_300`;
  verified first update and GPU allocation. Run`native_side_speed_300_20261028_232858_2935830`.

- User requested NO TEACHER and increased lateral speed reward. Implemented new
  task `lz_q1l_flat_native_side_speed`; old anchor task preserved for provenance.
- `tracking_lateral`1.5->3.0 only environment change vs native anchor config.
  Same exp(-vy_error^2/.04),other rewards unchanged. No teacher instance or
  imitation loss; frozen estimator,actor/critic/std PPO,fixedLR1e-5,newAdam.
- OriginalV8model200 initialization,4ms/decimation5,left/right50/50,no mirror.
  Stand/walk baseline unchanged. Config appended `_2_config.py`.
- Preflight passed32env100steps+one disposable update: teacher absent, estimator
  bitwise unchanged,actor updated,reward-only delta and native outputs verified.
  `media/native_side_speed_20261028/preflight_01/checks.json`.
- User subsequently authorized300updates/save50. Launcher
  `tools/launch_lz_q1l_native_side_speed.py` now300; preflight02 validates final
  code hashes. Actual launch/progress: `media/native_side_speed_20261028/latest_training.json`
  and its run/status.json. No automatic nextstage or deployment; no quality earlystop.
  Details `docs/lz_q1l_native_side_speed.md`.

## Native side model200 replay (2026-10-28)

- Completed200, all50 checkpoints. Replay pointer `media/native_side200_20261028/latest_result.json`.
- Continuous LEFT+.30/RIGHT-.30 each5s, own2s stand warmup,4ms/decimation5,
  native outputs both sides, NO MIRROR; strict hashes/config/action equality.
- No falls/resets; complete250frames/10s/full decode and HTTP200 verified.
- Last4s leftvy+.01650/rightvy-.18128m/s; height28.58/27.44cm.
  Left RH contact0%, force0N (LH/RH/LF/RF74.59/0/47.86/26.79N).
  Left yaw absolute mean36.97deg/s while signed mean nearzero: oscillation,
  not a steady turn. Native left remains unsuccessful; right below target.
- http://localhost:8769/native_side200_4ms_run_20261028_231558_RVMk/
- Single continuous nominal trial; no robustness/independent direction comparison.
  No new training/deployment. Stand/walk baseline unchanged.

## Native side-only anchored200 (2026-10-28, current training)

- Latest user: onlyLEFT/RIGHT50/50,200updates; explicitly NO MIRROR during
  training. Supersedes initial mirrored suggestion. New task
  `lz_q1l_flat_native_side_anchor` has native training, teacher and playback.
- ORIGINAL oldV8 model200 source,4ms/decimation5,|vy|.25-.45. Curriculum off;
  2048 environments fixedleft/2048 fixedright => equal rollout steps despite
  different reset/resampling counts. No stand/forward/yaw/mixed commands.
- Original rewards unchanged, no cadence penalty. Frozen old estimator and
  frozen native-direction source teacher. Action anchor coef1, mean squared
  error normalized by max(abs(source_std),.1); fixedLR1e-5,newAdam. Student
  actor/critic/std update; stdclamp.05..1. Teacher/estimator audited everyupdate.
  Native oldleft may be weak; never substitute mirror for its evaluation.
- Preflight02 final passed32env100steps +one disposable PPO update, likelihood
  consistency,teacher/estimator frozen,no mirror,balanced environment counts.
  Preflight01 is superseded sampling-count-balance variant.
- Detached tmux `lz_q1l_native_side_200`,GPU1. Pointer
  `media/native_side_20261028/latest_training.json`;run
  `logs2/flat_lz_q1l_native_side_anchor/native_side_200_20261028_225447_2911115`.
  Read status for actual progress. Save0/50/100/150/200;no quality early stop,
  no auto nextstep/deployment. Baseline and previous deployed mirror untouched.
- Details `docs/lz_q1l_native_side_anchor.md`,trainer
  `tools/train_lz_q1l_native_side.py`,algorithm`him_anchored_ppo.py` and dedicated
  `him_native_side_runner.py`. All newcfg/env appended in `_2` files.

## Original V8 at4ms video (2026-10-28)

- User requested un-finetuned simulation. Pinned ORIGINAL oldV8 model200 at
  4ms/decimation5, own2s stand warmup; continuous left+.30/right-.30 each5s,
  leftmirror,no resets. No training or modifications to source weights.
- Pointer `media/v8_original_4ms_20261028/latest_result.json`.
  http://localhost:8769/originalv8_4ms_4ms_run_20261028_223758_hDUM/
- Video last4s L+.407/R-.391m/s,height28.62/28.86cm. It overspeeds and drifts
  forward .111/.100m/s, so not perfect side tracking, but no19cm crouch.
  Different protocol from independent20s diagnosis; do not mix their numbers.

## 4ms regression isolated (2026-10-28)

- `media/v8_4ms_diagnosis_20261028/analysis.json`,findings_zh.md:30 matched
  independent20s side trials (5checkpoints x2directions x3starts),no resets.
  OriginalV8 at4ms: L.397/R.368m/s,height28.66cm (overspeed, not collapse).
  After50:L.059/R.041,height25.88;100:.018/.035,21.91;150:.068/.062,20.14;
  200:.089/.089,18.93. Serious regression arose in fine-tuning, not inevitable4ms.
- Inherited mixed commands only25% pure side; curriculum expanded vx1.0->1.5;
  actor/estimator both optimized, no old-action retention objective.
- Training21-50 vs151-200:weighted dofacc cost-1.821->-1.139 (largest positive
  change);height-.0056->-.0669;termination-.009->-.319. Tradeoff evidence,
  not causal ablation. Total return-19.48->-6.69 while length981->795;
  Episode terms per actual episode duration, Train return cumulative recent
  episodes: cannot sum window terms to explain exact total-return delta.
- No reward edits/new training. Recommended return to originalV8, constrain
  side-only adaptation and disable curriculum before advancing timestep.

## 4ms stage200 completed and replayed (2026-10-28)

- Training200 completed exit0,model50/100/150/200 saved. Pointer
  `media/v8_4ms_20261028/latest_training.json` completed. No nextstage launched.
- User requested simulation: `media/v8_4ms200_20261028/latest_result.json`.
  New model200 own standing warmup2s, continuous LEFT+.30 thenRIGHT-.30 each5s;
  left mirrors right,4ms/decimation5,noise/randomization off,no stage resets.
  Video http://localhost:8769/adapt4ms200_4ms_run_20261028_222843_KsCu/
- No falls/resets, but crouching and poor tracking: final4s segment means
  leftvy+.0869/rightvy-.0737 vs target+/-.30; baseheight18.93/19.06cm;
  allfeet contact95-99%. This is not successful4ms adaptation.
- Runtime config exactly checked against training before evaluation overrides;
  model state strict-loaded and checkpoint/source hashes checked.10s250frames
  complete ffmpeg decode passed. No baseline/model weights replaced.

## V8 4ms adaptation stage200 authorized (2026-10-28)

- Latest user approved firststage4ms/decimation5/20ms policy interval,200 updates.
  Start ORIGINAL oldV8 model200, not cadence150/200. Task `lz_q1l_flat_v8_4ms`.
  Config appended `_2_config.py`; inherit oldV8 rewards/commands, no cadence cost.
  Restore model and both optimizers; fixed PPO LR1e-5. Four delay choices mapped
  to0/4/8/16ms on4ms grid. Baseline stand/walk unchanged; left playback mirrored.
- Preflight32env160steps passed; frozen baseline hash and exact source load
  checked. Save0 and50/100/150/200 new updates; no automatic next timestep stage
  or quality early stop. Details `docs/lz_q1l_v8_4ms.md`.
- Pointer `media/v8_4ms_20261028/latest_training.json`; tmux
  `lz_q1l_v8_4ms_200`; selected GPU2 UUID13617fc8... . Read runstatus for progress.
  Run `logs2/flat_lz_q1l_v8_4ms/v8_4ms_200_20261028_210023_2842293`.
- Previous cadence200 completed normally. Cadence150 evaluation/report at
  `media/lateral_cycle150_20261028/analysis.json` / findings_zh.md,12 trials no
  resets. Same5ms originalV8 vs150 side speeds L .307->.246,R .280->.138;
  no uniform frequency improvement, front loaded-leg cycles worsen.
  Video http://localhost:8769/cadence150_5ms_run_20261028_000259_Q7Va/
- User discussed continuous mixed1ms/5ms video, then moved to adapting side to
  uniform1ms via stages. Mixed-timestep video was NOT implemented or recorded.

## Active lateral0.8s cadence continuation200 (2026-10-27)

- User authorized side period0.8s, everything else unchanged,200 updates.
  New task `lz_q1l_flat_lateral_cycle`, cfg/env appended to `_2_config.py`/`_2.py`.
  Inherits OLD V8 entire config (5ms/decimation4,4096envs,rollout100,old commands
  and native left/right training); only additional lateral_cycle reward-.5.
  Playback left still mirrors right. New stand/forward baseline remains separate.
- Source oldV8 model200 SHA3480ca2abddfcd62df2be30d89e53a992e3eadc100add59df131213feef48a5a.
  Network and both optimizer states restored exactly, saved LR1e-5 restored.
  New iteration numbering0..200;save0 plus50/100/150/200. No quality early stop.
- Target eachfoot valid landing interval.8s +/- .1s; validstep requires prior
  stable contact, >=.06s air with >=1cm collision clearance, then>=.04s >=5N.
  Held completed-period error plus overdue penalty prevent no-step evasion;
  short fake contact/clearance cannot refresh clock. Non-side reward zero.
- Preflight `media/lateral_cycle_20261027/preflight_01/checks.json` passed
  synthetic edgecases and32env160stepGPU checks;configdiff only cadence rewards.
- Actually launched onGPU1 UUID33c8a561...,PID2192786,tmux
  `lz_q1l_lateral_cycle_200`; verified running2+updates/GPU9108MiB.
  Pointer `media/lateral_cycle_20261027/latest_training.json` and runstatus
  `logs2/flat_lz_q1l_lateral_cycle/v8_cycle080_200_20261027_231332_2192772/status.json`.
  Consult live status, not this historical launch note. ~4s/update at launch.
- Details `docs/lz_q1l_lateral_cycle_080.md`;trainer
  `tools/train_lz_q1l_lateral_cycle.py`;detached launcher
  `tools/launch_lz_q1l_lateral_cycle.py`. No automatic deployment/1ms transfer.

## V8 5ms vs 1ms side test (2026-10-27)

- User requested restore original frequency and record each side5s. Diagnostic
  overrides only; production baseline/composite remain1ms. No training.
- Initial whole-composite5ms test failed during NEW baseline stand warmup(step67)
  and left(step12); failed run retained `media/v8_timestep_compare_20261027/
  run_20261027_225841_T0mh`. Do not claim newbaseline works at5ms.
- Isolated V8 comparison uses OLD V8 own standing warmup2s, continuous LEFT+.30
  thenRIGHT-.30 each5s; same pinned weights, leftmirror, no noise/randomization.
  Both5ms/decimation4 and1ms/decimation20 are50Hz policy. Neither pure run reset.
- Statistics last4s:5ms left+.307/right-.285;1ms left+.168/right-.213 m/s.
  This supports timestep/PD-dynamics sensitivity; not perfect posture or robust
  convergence proof. Warmup evolves under the selected physics in both cases.
- `media/v8_timestep_compare_20261027/result_5ms.json`,result_1ms.json,
  comparison.json and findings_zh.md. record_pure.sh takes5 or1 argument.
  Active config differences dt/decimation; action delay and push disabled.
- Requested5ms video: http://localhost:8769/v8_side_5ms_run_20261027_230007_Cotw/
  Matched1ms video: http://localhost:8769/v8_side_1ms_run_20261027_230042_9igj/

## Baseline + legacy V8 lateral composite (2026-10-27)

- User authorized old V8 model200 side policy, LEFT mirrored from RIGHT,
  while retaining accepted new baseline stand/forward and ALL 1ms physics.
- New inference-only task `lz_q1l_flat_baseline_v8_lateral`; config appended
  to `_2_config.py`, routing `utils/lz_q1l_legacy_lateral.py`, dedicated runner
  `him_legacy_lateral_runner.py`. Hash-pinned800 and oldV8-200, both frozen.
  No training; old V4 direct trained model retained as separate task.
- Baseline config unchanged, dt.001/decimation20; oldV8 own estimator;
  physical history shared, only pure-left V8 input/output reflected.
  Side switches immediate, inherited stand/forward RH blend .3s unchanged.
- Video `media/baseline_v8_lateral_20261027/latest_result.json`,40s:
  stand/forward.35/left.30/right.30/stop,8s each, no stage resets or falls.
  http://localhost:8769/baseline_v8_lateral_run_20261027_225233_H3VI/
- Every frame checked nonlateral outputs exactly match baseline and side outputs
  match oldV8 wrapper. Full ffmpeg decode/1000frames/40s checks passed.
- Last5s left/right vy .220/-.118 vs target +/- .30, roll5.52/2.97deg;
  loads LH/RH/LF/RF left70.6/7.9/51.3/21.5N, right13.2/69.7/22.3/44.0N.
  Transfer is functional, not proof of good side gait or old5ms equivalence.
  Initialstand andstop allfeet100%contact, stoprear43.0/42.9N in this one run.
- Analysis `media/baseline_v8_lateral_20261027/analysis.json`; details
  `docs/lz_q1l_baseline_v8_lateral.md`. Single continuous nominal trial;
  different physical histories confound left/right speed comparison.

## V4 convergence log audit (2026-10-27)

- `media/v4_convergence_20261027/findings_zh.md`, `analysis.json`, convergence.png.
 500 updates finite. Native TensorBoard steps +1; V4/Directions already completed
 updates. Training windows include exploration/randomization/transitions.
- Around300 reward/tracking plateau:301-350 reward38.0547 ->451-500 37.8473;
 left side MAE .04932->.04992, right .05467->.05481m/s. Not full convergence.
- Support worsens: left RH force22.79->19.90N/contact29.83->25.03%; right LF
 force10.58->6.11N/contact25.70->13.45%. Orientation penalty magnitude +42%.
- V4/lateral_std .279->.292; generic Policy/mean_noise_std logs frozen baseline
 std and must not be interpreted as lateral exploration. PPO loss/KL not success.
- No new training. Intermediate300/400/500 standardized comparison remains a
 proposed next check, not performed; do not claim model300 is best from reward.

## V4 model500 direct simulation and analysis completed (2026-10-27)

- Video pointer `media/v4_500_20261027/latest_result.json`, continuous8s each:
  stand / forward .35 / left .2 / right .2; no mirror, no stage resets, no falls.
  http://localhost:8769/new_v4_500_run_20261027_220757_jCJg/
- Model500 SHA256 e985f5b84ef44acfbc90fe56f31cf1f84bb2527a38222cabfdbf5c6a94f32319.
- Supplement:30 independent20s command trials,3 root-velocity starts each,
  noise/randomization off. V4 side speeds .15/.2/.25 plus stand/forward, and
  frozen-baseline stand/forward controls. No failures. Not training seeds.
- Side tracking remains poor: left actual .0527/.1286/.1976, right magnitude
  .0772/.1336/.1822. At .2 forward drift left .0828/right .0358m/s.
- At .2 left force LH/RH/LF/RF=78.3/12.4/43.6/21.2N, right35.7/49.2/6.4/60.6N.
  Left RH contact21.9%; right LF17.7%. Height left26.44/right30.85cm.
  Left RF contact slip .124m/s; right LF .087m/s. Gait is not yet coordinated.
- Stand/forward preserved (same-input exact,30 frozen tensors unchanged).
  Findings `media/v4_500_20261027/findings_zh.md`; full `analysis.json`.
  No new training or reward changes. Model50-450 not evaluated this turn.

## V4 direct lateral500 completed (2026-10-27 run)

- Latest status check: completed500/500, normal exit0. All50-step checkpoints
  model50 through model500 present. Final frozen audit:30 baseline tensors unchanged.
- Authoritative pointer `media/v4_lateral_20261027/latest_training.json` now completed.
  No further training launched. Training completion does not establish lateral
  tracking quality; use direct unmirrored V4 inference for subsequent evaluation.

## Active V4: DIRECT left/right, 500 authorized updates (2026-10-27)

- User rejected mirrored training. Keep stand20%, forward20%, lateral60%, with
  left/right each half WITHIN lateral commands (overall30/30). V4 training and
  replay use real observations/actions without mirroring or joint permutation.
  One command-conditioned lateral branch learns both signs; no forced symmetry.
- New V4 sampler balances lateral command counts, alternating the extra sign
  for odd batches. Counts differ by at most1; reset timing can change exposure.
- Source remains accepted NEW composite baseline (gait150 stand/model800 walk).
  Actor/estimator/RH correction frozen. Rewards, speeds .15-.25, LR3e-4 unchanged.
- User authorized500 new updates, save50. Session `lz_q1l_v4_lateral_500`.
  Authoritative pointer `media/v4_lateral_20261027/latest_training.json`, run
  `logs2/flat_lz_q1l_v4_lateral/direct_from_newbaseline_500_20261027_200558_2064625`.
  Read its status/log for real progress; do not infer completion from this note.
- Direct variant checks: `media/v4_lateral_review_20261027/direct_no_mirror/`;
  CPU no-mirror/likelihood/freeze checks and GPU64x160 passed. Never use initial
  mirrored proposal/checks in the parent review directory as current evidence.
- No quality early stop. Detached job survives remote disconnect. Old model
  mirroring defaults remain unchanged; V4 is the user-authorized exception.

## New baseline accepted; ORIGINAL V4 review (superseded by direct variant above)

- User explicitly accepted the current composite as NEW baseline: gait150 stand,
  model800 forward, .30s RH blend. Pinned copies/manifest/source/evidence in
  `HIMLoco/legged_gym/logs2/baselines/stand_walk800_20261027/`. This is not oldV8
  and not raw model800 standing. Preserve its known stop-load/high-speed limits.
- New task `lz_q1l_flat_v4_lateral`, config/environment appended in `_2_config.py`
  and `_2.py`. Read `docs/lz_q1l_new_v4_lateral_review.md` before work.
- Frozen baseline actor/estimator/RH correction; only independent12-output side
  residual, side exploration and critic train. Both physical lateral directions
  share canonical-right mirroring during rollout AND PPO update; old V7/V8
  weights and old LH-to-RH teacher are not used. No turn/diagonal training yet.
- Commands stand20/forward20/left30/right30%, |vy| .15-.25, fixed LR3e-4.
  OldV7 tracking1.5 and V8 lateral_error-.5, side swing/sliding-.2 each.
  Height/orientation/load weights unchanged. Side mode switches are not smoothed;
  the accepted .3s blend still covers stand/forward only.
- CPU gradient/isolation/mirror/density checks passed. GPU64x160 steps finite,
  zero resets, correct reward gates. This proves wiring, not learned sidestepping.
  Results `media/v4_lateral_review_20261027/checks.json`.
- Formal training NOT authorized/launched. Default iterations0; launcher
  `tools/train_lz_q1l_v4_lateral.py` requires explicit count/GPU/newrun. Save50.
  Dedicated runner is selected by optional runner_class_name in task_registry;
  legacy HIMOnPolicyRunner is unchanged. Never confuse old `lz_q1l_flat_v4`.

## Continuous speed sequence recorded/analyzed (2026-10-27)

- Latest requested video: baseline stand8, model800 forward .2/.4/.5 m/s x8s,
  continuous no stage resets. `media/v3_speed_sequence_20261027/latest_result.json`.
- Video and9 matched comparison trials completed without resets. Last5s stats:
  hybrid vx .214/.455/.610; height28.62/27.39/26.09cm; absroll .54/1.86/.63deg.
  At .4 mean-force gaps rear/front .35/.55N vs baseline5.91/6.61N.
  At .5 gaps4.64/3.02N, vxMAE .115 vs baseline .036; about22% mean overspeed.
- Better load balance does not establish symmetric gait timing: at .4 video
  LH/RH contact50.4/62.0%. Walking swing is not abnormal suspended-foot failure.
- Analysis `media/v3_speed_sequence_20261027/analysis_zh.md` and `analysis.json`.
  Sequential speeds confound speed with history. No new training or reward edits.

## Baseline standing / model800 walking simulation (2026-10-27)

- User-authorized inference-only hybrid implemented as `lz_q1l_flat_stand_walk`.
  Uses frozen gait150 baseline for standing and split model800 for forward;
  RH residual ramps over .30s (15 control steps). No new training.
- Configuration and environment appended in `_2_config.py` and `_2.py`;
  adapter is `utils/lz_q1l_stand_walk.py`. Same-input other9 outputs untouched;
  standing/forward endpoints exactly match baseline/model800 respectively.
- Video pointer `media/v3_stand_walk_20261027/latest_result.json`;
  findings `media/v3_stand_walk_20261027/findings_zh.md`.
  Continuous stand8/forward8/stop8/restart8, no stage resets. Matched evaluation
  of baseline/model800/hybrid across3 initial states:9 trials, no resets.
- Walking retains model800 balance, but stopping does NOT recover initial
  standing load: hybrid LH75.18/RH14.67N, left64.82%, despite all feet touching.
  Pure baseline also shows LH80.23/RH11.28N after walking and stopping.
  Do not claim switching solves standing recovery, or hide it with stage resets.

## Split-weight800 completed, recorded and analyzed (2026-10-27)

- User-authorized new500 from RH-only300 is complete: cumulative800. Frozen
  policy22 tensors unchanged. Final hash
  `23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086`.
- Single video `media/v3_split800_20261027/latest_result.json` (stand8/forward8).
  Comparison video/report `media/v3_split800_analysis_20261027/latest_result.json`:
  http://localhost:8769/new_v3_split800_comparison_20261027/
- Fresh baseline/300/800 replay: 36 trials, no resets. At .35m/s, 300->800:
  roll2.89->1.87deg, rear mean-force gap4.56->.39N, front10.74->.36N,
  left share54.68->49.99%. Both pairs improve, not simple gap transfer.
- Same baseline histories: teacher target-angle-equivalent RMSE2.80->1.73deg;
  own histories2.98->2.19deg. RH correction grows; thigh/calf target asymmetry
  shrinks, but roll-joint asymmetry grows. Common-state and own-state metrics
  are separate in teacher_analysis.json; teacher match is not posture match.
- Standing worsens: LF55.68/RF24.24/LH19.93/RH49.94N; left share50.48% hides
  diagonal imbalance. Rear/front gaps30.01/31.44N, roll2.73deg, height33.91cm.
  At .35 velocity MAE .0349->.0491m/s, RH slip .0120->.0214m/s. At .5 commanded,
  vx=.609 and height26.11cm. Keep800 as walking candidate, preserve baseline.
- Extra500 updates and forward coefficient increase are confounded: no equal-
  budget .1 continuation control. No new training launched by this analysis.

## Active continuation extended to cumulative800 (2026-10-27)

- Latest user: "那就续训500轮". Replaces the total new300 request with new500
  from RH-only300: cumulative800, same stand.1/forward.3, save every10.
- The active300->600 run remains uninterrupted. A detached queue waits for its
  successful final600 checkpoint, exit0 and frozen audit, then restores all
  weights/optimizer states/saved LR and trains200 more to800 in a fresh run.
- Overall authoritative pointer:
  `media/v3_right_rear_split_20261027/continuation_800.json` (target800).
  `latest_training.json` in that folder still describes only the first stage
  to600; do not mistake its completion for the overall request being complete.
- Queue session `lz_q1l_v3_right_rear_split_to800`, launcher
  `tools/start_lz_q1l_v3_right_rear_split_800.sh`, trainer
  `tools/train_lz_q1l_v3_right_rear_split_extend.py`. It does not restart failed
  or user-stopped upstream training. Future user STOP must stop both the active
  training and this queue, to prevent unintended continuation.
- Current-stage run and all source hashes are unchanged. Extension code is
  separate and snapshotted; the queue verifies source hashes and model600 digest
  before launch. Overall progress counts additional_completed=iteration-300.

## Active RH-only continuation: stand .1 / forward .3, 300 more (2026-10-27)

- Latest user instructions resolve to standing imitation .1, forward imitation
  .3, and 300 additional updates. Supersedes the earlier 200-update request and
  provisional .2 coefficient. Resume fixed RH-only model300 to cumulative600.
- Source SHA256 `8d41715f4f7836aafd15356ee0593ac68d879268b9384774a69c70084b68d97b`;
  teacher remains original gait150. Restore all student weights, both optimizer
  states and saved adaptive LR .01. Frozen baseline actor/estimator/std remain
  fixed. Do not reset the RH correction or train other-leg policy outputs.
- New task `lz_q1l_flat_v3_right_rear_split`, config appended to
  `lz_q1l_2_config.py`; separate `HIMRightRearSplitPPO` changes only per-command
  teacher weights. Environment/rewards, 4096 envs, rollout100, seed1 unchanged.
  Samples retain the same mean reduction; forward teacher gradients3x, standing
  unchanged. Additional logs separate standing/forward MSE and sample counts.
- Save cumulative model310,320,...600 in a fresh logs2 directory. No quality
  early stop. Background session `lz_q1l_v3_right_rear_split_600`; launcher
  `tools/start_lz_q1l_v3_right_rear_split_600.sh`; latest status pointer
  `media/v3_right_rear_split_20261027/latest_training.json`.
- Preflight checked equal-weight equivalence, 3x forward gradient, no teacher
  gradient to other9 outputs/other commands, unchanged environment and intended
  config-only differences. Startup resume_audit verifies exact weights/moments.
- Old replay/training entrypoints pin previous code hashes; source archives
  remain intact. Do not weaken old hashes to misrepresent compatibility.

## Three-model coordination comparison (2026-10-27)

- User requested baseline / RH-only150 / RH-only300 side-by-side comparison.
  Three-column video (same command time, not phase aligned) and fresh 36-trial
  evaluation with body-frame foot trajectories are in
  `media/v3_coordination_comparison_20261027/latest_result.json`.
  http://localhost:8769/new_v3_baseline_150_300_coordination_20261027/
- At .35m/s all three RH cycles are .34s, RH lands about40ms after LF.
  This is unchanged across 1/5/10N contact thresholds at 50Hz resolution.
  Do not claim progressive synchronization merely from improved appearance.
- Touchdown-aligned, mean-removed LH/RH x-z cycle RMS difference is
  11.53/18.51/16.75mm: final300 improves this measure over150, not overbaseline.
  Roll improves, load balance favors150; no universal coordination winner.
  Timing/trajectory criteria and limitations are in gait_analysis.json and
  findings_zh.md. No controller or training changes made.

## RH-only v3 final300 recorded and compared (2026-10-27)

- Authorized 300 updates completed; final frozen-policy audit passed (22 tensors
  bitwise unchanged). No additional training authorized by the replay request.
- Video: `media/v3_rh_only300_20261027/latest_result.json`, stand8s + forward8s,
  http://localhost:8769/new_v3_rh_only300_run_20261027_163911_FtQk/
- Fresh matched baseline150/RH-only150/RH-only300 evaluation: 4 commands x
  3 starts x 20 seconds each, 36 trials, no resets. Results and limitations in
  `media/v3_rh_only300_analysis_20261027/findings_zh.md`.
- At .35m/s, roll 3.49 -> 3.16 -> 2.89 deg, rear mean-force gap
  6.63 -> 3.23 -> 4.48 N, front gap 8.14 -> 9.76 -> 10.70 N;
  left load share 54.59 -> 53.97 -> 54.64%. Final300 improves roll but loses
  the overall load improvement at150. Standing final300 RH51.70N/RF17.59N,
  all-four contact100% hides marked force imbalance. Keep original baseline.
- Standing load varies between replay batches; do not mix historical values
  into fresh comparisons or mistake three replay starts for training seeds.

## RH-only v3: 300 new updates authorized (2026-10-27)

- Latest user request: "那你训300轮看看". Authorizes the reviewed RH-only model
  below for 300 updates, superseding its review hold. Does NOT resume old sweep.
- Start from pinned gait150; frozen baseline actor/estimator/std and LH teacher;
  train only RH three-output residual plus separate critic. Teacher weight .10,
  no other-leg soft penalty; baseline environment/rewards unchanged. Fresh Adam
  for the changed trainable parameter set, baseline saved LR ~2.25e-5.
- Keep 4096 envs, 100 rollout steps, seed1, save every10 through model300.
  No quality-based early stop. Select GPU with greatest free memory, >=12288 MiB.
- Launcher `tools/start_lz_q1l_v3_right_rear_only_300.sh`; tmux
  `lz_q1l_v3_right_rear_only_300`; pointer
  `media/v3_right_rear_only_20261027/latest_training.json`.
  Run status.json is the source of actual completed updates. At completion,
  frozen_policy_audit.json checks baseline actor/estimator/std remain bitwise fixed.

## RH-only v3 correction: implemented for REVIEW, no training (2026-10-27)

- User clarified repeatedly: only RIGHT HIND should learn LEFT HIND; keep the
  front legs and left hind policy fixed. Old soft-preservation sweep is stopped.
- New task `lz_q1l_flat_v3_right_rear_only`, config classes appended in
  `lz_q1l_2_config.py`; baseline environment/rewards exactly unchanged.
  `HIMRightRearOnlyActorCritic` freezes gait150 actor, estimator and all action
  std parameters. An independent zero-initialized three-output residual can
  alter ONLY named RH joints. Other nine outputs are structurally unchanged
  for identical histories. No other-leg soft penalty; RH teacher coefficient .10.
- Frozen gait150 LH teacher queried on mirrored full histories, as before.
  Only RH branch and separate critic are optimizer parameters. Estimator update
  is never called. Fresh optimizer required for changed trainable parameters;
  keep gait150 weights and its saved LR. Do not restore full-body Adam moments.
- `tools/train_lz_q1l_v3_right_rear_only.py` requires explicit iterations/GPU/new
  directory. Default max_iterations=0. NO training authorized or started by this
  clarification. Previous STOP remains in force until a new training command.
- Review `docs/lz_q1l_v3_right_rear_only_review.md`; validation
  `media/v3_right_rear_only_review_20261027/report.json`: zero updates, other9
  exact-output invariance under temporary RH branch perturbation, gradients,
  frozen 6-env/250-step replay, backward-only integration and checkpoint loading.
- Fixed other-leg POLICY does not promise fixed physical force or trajectories
  under changed body states; disclose this distinction and continue whole-body
  load/posture evaluation after any later authorized training.

## Five available v3 checkpoints analyzed after STOP (2026-10-27)

- User subsequently requested comparing every group to baseline. Only frozen
  0.03/model270 replay was added; no training resumed. Other four final300 and
  baseline evaluations reused after checkpoint/evaluation-source hash checks.
- Results: `media/v3_all_analysis_20261027/findings_zh.md`, `comparison.json`,
  `comparison.png`. 72 matched trials total, no resets. 0.03 uses 270 updates,
  other v3 groups 300; do not claim a controlled equal-budget optimum.
- 0.03/model270 is a balanced candidate, not a replacement baseline: at .35 m/s
  roll=3.59 vs baseline3.52 deg, rear gap=5.28 vs7.14 N, front gap=9.40 vs7.28 N.
  Standing roll=.46 deg, left share50.82%, but rear gap14.35 vs5.99 N and front
  gap16.82 vs9.25 N. Standing RH51.35 N / RF22.30 N: near50% total side share
  hides front/rear pair imbalance. No model uniformly improves all key metrics.
- Keep original gait150 baseline. Training stop below remains in force.

## User STOP: v3 sweep terminated (2026-10-27)

- Latest user command: "别训了 停止训练". This supersedes the remaining
  300-update authorization below. Do not restart or resume without a new command.
- Scheduler PID 1145486 stopped; remaining 0.03 training PID 1145494 exited.
  That group completed 279 updates; latest saved checkpoint is model_270.pt.
  The other four groups completed 300 and their model_300.pt files are preserved.
- Automatic full-sweep evaluation is cancelled. Existing four-group evaluation,
  videos, all checkpoints and logs remain intact. Pointer and manifest are marked
  stopped_by_user with auto_resume=false.

## Four completed v3 groups evaluated (2026-10-27 morning)

- User requested analyzing completed 0/0.01/0.02/0.04 final300 models and recording
  the best candidate while 0.03 continues. Independent evaluation is in
  `media/v3_four_comparison_20261027/evaluation/`; findings `findings_zh.md` in
  its parent. Same baseline, 12-env/3-start/4-command/20s protocol; 60 trials,
  no resets. Do not mistake this partial report for the full sweep's final report.
- Selected 0.01 for WALKING-oriented video: lowest roll and rear mean-load gap
  among the four new models at all three speeds. It is NOT uniformly better
  than baseline. At 0.35 m/s, rear gap 7.14 -> 1.55 N, but roll 3.52 -> 4.15 deg
  and front gap 7.28 -> 13.19 N. Standing front gap 9.25 -> 42.51 N; right front
  carries ~10 N vs left front ~52.5 N despite all-four contact. Preserve baseline.
- Video `media/v3_four_comparison_20261027/latest_result.json`: model300 keep=.01,
  8s stand + 8s forward(.35), nominal independent starts, no resets.
  http://localhost:8769/new_v3_selected300_run_20261027_073808_pjKN/
  No new training changes authorized by this analysis request.

## New v3 five-weight sweep: 300 updates EACH authorized (2026-10-26)

- Latest user instruction authorizes five equally spaced preservation weights in
  parallel, and corrects the duration from 500 to **300 updates per group**.
  This supersedes the earlier review-only hold below for these five experiments.
- Scan `other_legs_coef = 0, 0.01, 0.02, 0.03, 0.04`; keep RH teacher coefficient
  0.10. Each starts independently from original gait150, restoring both optimizer
  states and saved learning rate. Preserve baseline environment and all rewards,
  4096 environments, 100 rollout steps, seed 1. Save every 10, no quality stop.
- Host has four GPUs; run at most one of our training jobs on each GPU (sharing
  with existing unrelated jobs is authorized). Select greatest available memory
  with at least 12288 MiB. Four jobs can run concurrently; fifth starts when a
  slot is released. Do not reduce environment count to force five concurrent jobs.
- Launcher `tools/start_lz_q1l_v3_sweep.sh --iterations 300`, tmux
  `lz_q1l_v3_sweep_300`; pointer `media/v3_sweep_20261026/latest_training.json`.
  Individual job `status.json` reports actual completed updates. Manifest has
  coefficients, GPU UUIDs, source digests, PIDs and separate run directories.
- Only startup CUDA OOM before any update/checkpoint may retry (at most three
  retries, with higher free-memory requirement); retain all failed attempts.
  Other failures are recorded, not silently restarted or stopped by quality.
- After all jobs finish, automatically evaluate baseline150 and each successful
  final300 separately under the same 12-env, 3-start, 4-command, 20-second protocol.
  Compare posture/load/gait/front-leg effects; no automatic winner by total reward.
- Validation: `media/v3_rear_teacher_sweep_review_20261026/report.json`.
  Experiment plan: `docs/lz_q1l_v3_sweep_plan.md`.

## New v3 rear-leg teacher: original review record (2026-10-26)

- Original instruction: implement as new v3, deliver for review, and WAIT for
  a subsequent explicit training command. This hold was superseded by the five
  300-update experiments above; other unrequested training remains unauthorized.
- Task `lz_q1l_flat_v3_rear_teacher`; review `docs/lz_q1l_new_v3_review.md`;
  validation `media/v3_rear_teacher_review_20261026/report.json` (zero updates).
- Student and frozen teacher both originate from the pinned original gait150
  below. Mirror all six observation frames, query the raw frozen teacher, mirror
  actions back, supervise only RH three outputs (MSE coefficient 0.10). Preserve
  other nine outputs with the teacher's original-state actions (coefficient 0.02).
  Stand/straight only. Baseline environment/rewards unchanged; no later load cost.
- Configuration appended to `lz_q1l_2_config.py`; algorithm is
  `HIMRearTeacherPPO`; explicit entry `tools/train_lz_q1l_v3_rear_teacher.py`.
  Reuse existing gait environment. No duration was authorized at this review stage.
  Existing completed experiments below remain historical, not launch commands.

## Current baseline: original gait150 (user decision, 2026-10-26)

- The user designated the ORIGINAL gait-recovery model_150 as the baseline for
  future changes. This decision supersedes the historical continuation sources
  below. Machine-readable pointer: `HIMLoco/legged_gym/logs2/baseline.json`.
- Checkpoint (relative to HIMLoco/legged_gym):
  `logs2/flat_lz_q1l_v2_gait/gait_from_posture500_200_20261025_113614_3235946/model_150.pt`.
  Verified SHA256: `655a5022f2e1404c8f7c3dfbb670cc8235710f78e7550e72f2343e4e2c5e9747`.
- Start future experimental branches from this checkpoint and its original gait
  configuration (`LZQ1LFlatV2GaitCfg` / PPO, `LZQ1LV2GaitRobot`), applying only
  the changes requested for that experiment. Use the source run's env_config.json,
  train_config.json and provenance as the configuration reference. Do not silently
  inherit later stand-load, height, upright or balance-roll changes, or select a
  checkpoint by latest timestamp. In particular, the completed stand-load200 is
  an experiment result, not the baseline; other runs' model_150 are not this model.
- Preserve the baseline checkpoint/configuration snapshots and write new runs to
  separate logs2 directories. Compare future results against this fixed baseline
  under matched evaluation conditions. Unless explicitly changed, continuation
  restores both optimizer states and the saved PPO learning rate as before.
- This baseline selection does not itself launch another training run. The
  standing-load200 run below is completed and is retained as historical evidence.

## Stronger LEFT-load continuation (2026-10-26; 300 updates authorized)

- The user corrected the preceding request: greatly increase LEFT OVERLOAD,
  not posture. The unlaunched stronger-posture candidate has been replaced by
  `lz_q1l_flat_v2_side_load_strong`, `LZQ1LFlatV2SideLoadStrongCfg` / PPO,
  using the same `LZQ1LV2SideLoadRobot`. Sole environment change versus side-load:
  left_load_excess -1.0 -> -10.0 (10x). Preserve orientation=-0.2, height=-10,
  and all other settings. Formula/window unchanged; right-heavy states incur
  exactly zero NEW cost. Standing and straight walking both covered.
- The user explicitly authorized 300 NEW updates from original pinned gait150,
  NOT regressed side-load100. Restore both optimizers and saved PPO learning rate;
  preserve 4096 environments, 100 rollout steps, seed 1 and all baseline settings
  except the added left-only load cost/window. Save every 10 through model_300.
  Historical configurations remain intact; no quality-based early stopping.
- Launcher: `tools/start_lz_q1l_v2_side_load_strong_300.sh`; tmux
  `lz_q1l_v2_side_load_strong_300`; pointer
  `media/v2_side_load_strong_20261026/latest_training.json`. Run outputs under
  `HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_side_load_strong`.
  Validation: `media/side_load_strong_validation_20261026/report.json`.
- After training, automatically compare baseline150 against new100/200/300 with
  the same matched 12-env/20-second protocol via `tools/eval_lz_q1l_side_load_strong.py`.
  Reports in the run's `evaluation/`; evaluation failures do not alter training.
  Existing non-strong side-load launchers remain for the completed old experiment.

## One-sided left-load experiment (completed, 2026-10-26)

- The user requested a new penalty and training from the fixed gait150 baseline,
  then explicitly corrected the design: penalize ONLY left-side total load greater
  than right-side total load; no new penalty when the right side is heavier.
- Task `lz_q1l_flat_v2_side_load`: `LZQ1LV2SideLoadRobot`,
  `LZQ1LFlatV2SideLoadCfg` / PPO, directly inheriting original gait classes.
  New term `left_load_excess=-1.0`: square(max(0, L-R)/(L+R+1 N)), using
  1-second rolling mean of each side's summed nonnegative foot Fz. No deadband,
  no symmetric penalty, no low-total extra cost. Stand and straight forward only;
  warm up a full window after reset/resampling/command changes. Other baseline
  penalties remain unchanged (including the symmetric rear-pair term).
- User authorized training but did not specify duration; agent stated the initial
  choice of 100 additional updates, saving every 10, after asking asynchronously.
  Restore weights, both optimizers and saved PPO LR from pinned gait150. Keep
  4096 envs, 100 rollout steps, seed 1, height/orientation and command mix unchanged.
  No quality-based early stopping. This experiment does not replace the baseline.
- Launcher: `tools/start_lz_q1l_v2_side_load_100.sh`; tmux
  `lz_q1l_v2_side_load_100`; pointer `media/v2_side_load_20261026/latest_training.json`.
  Validation: `media/side_load_check_20261026/report.json` (one-sided formula,
  named-foot grouping, history lifecycle, inherited gait rules, optimizer restore,
  frozen-policy six-env CPU smoke with no resets).
- After successful training, queue automatically runs `tools/eval_lz_q1l_side_load.py`
  comparing baseline150, new50, new100 independently with the same 12-env layout,
  3 paired starts and 20-second standing / 0.2 / 0.35 / 0.5 m/s commands. Results
  in the run's `evaluation/`; evaluation failure is separate from training status.
  Judge absolute deviation from 50/50, including individual starts, and roll/gait;
  lower left-only cost alone can hide a shift to right-heavy support.

## Historical standing-load continuation (completed, 2026-10-26)

- After reviewing the gait150 video, the user authorized 200 new updates from
  `logs2/flat_lz_q1l_v2_gait/gait_from_posture500_200_20261025_113614_3235946/model_150.pt`.
  Source SHA256: `655a5022f2e1404c8f7c3dfbb670cc8235710f78e7550e72f2343e4e2c5e9747`.
- New task `lz_q1l_flat_v2_stand_load` uses `LZQ1LFlatV2StandLoadCfg` / PPO in
  `lz_q1l_2_config.py`, directly inheriting the gait configuration. Reuse
  `LZQ1LV2GaitHeightRobot` from `lz_q1l_2.py` for its standing-load methods;
  that environment class does not itself change the height target or reward.
- Add only `stand_pair_force_balance=-0.5` and `stand_low_load=-0.2`, with
  0.5 s standing window/grace, 10% pair tolerance and 10% minimum foot share.
  Both new penalties are disabled while walking. Keep base_height=-10,
  orientation=-0.2, the three gait terms, physics, PD, command mixture and PPO
  settings from gait150. Do not inherit later height/upright/balance-roll scales.
- Restore both optimizers and saved PPO LR; restart only the stage counter and
  simulator/RNG. Train 200 additional updates, save every 10 (10 through 200),
  4096 environments and 100 rollout steps. No quality-based early termination.
- Launcher: `tools/start_lz_q1l_v2_stand_load_200.sh`; dedicated tmux session
  `lz_q1l_v2_stand_load_200`; pointer `media/v2_stand_load_20261026/latest_training.json`.
  Outputs: `HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_stand_load`.
  Actual progress is in the pointed run's `status.json`, not the queue state.
- Validation: `tools/check_lz_q1l_stand_load.py` and
  `media/stand_load_check_20261026/report.json`: configuration-delta audit,
  reward/lifecycle invariants, full optimizer restore, and six-environment
  frozen-policy CPU smoke passed; walking standing-load costs are zero.

## Configuration layout

- Keep the original V1--V8 configurations in
  `HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py`.
- New V1 and all subsequent new-lineage versions belong in
  `HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py`.
  Append environment and PPO configuration classes using inheritance, following
  the original file's organization; do not create a separate config file per version.
- The shared 1 ms classes `LZQ1LFlatV1OneMsCfg` / `LZQ1LFlatV1OneMsCfgPPO`
  also live in `lz_q1l_2_config.py`; the former standalone
  `lz_q1l_v1_1ms_config.py` has been merged into it. Keep the existing
  `lz_q1l_flat_v1_1ms` task registered for timestep diagnostics.
- The new V1 task and class names remain `lz_q1l_flat_v1_support` and
  `LZQ1LFlatV1SupportCfg` / `LZQ1LFlatV1SupportCfgPPO`.
- New-lineage environment logic belongs in
  `HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py`, paired with
  `lz_q1l_2_config.py`. The new V1 environment class remains
  `LZQ1LV1SupportRobot`; extend this module for subsequent versions.
  The original `lz_q1l.py` remains the old-lineage environment.
- New V2 is registered as `lz_q1l_flat_v2_unified` using
  `LZQ1LV2UnifiedRobot` and `LZQ1LFlatV2UnifiedCfg` / `LZQ1LFlatV2UnifiedCfgPPO`.
  Do not confuse it with old `lz_q1l_flat_v2`. Its dedicated entrypoint is
  `tools/train_lz_q1l_v2_unified.py`, which loads the pinned new V1 model_500
  weights with fresh optimizers and requires explicit iterations/GPU/run directory.
  Review details: `docs/lz_q1l_new_v2_review.md`. After reviewing the implementation,
  the user authorized 800 new V2 updates on 2026-10-24, saving every 50 updates.
  Choose the GPU with greatest free memory, as requested, even if already shared.
  The user explicitly stopped this run on 2026-10-24 after the posture regression
  diagnosis. It exited after 338 completed updates; latest saved checkpoint is
  model_300.pt. Do not restart this run without a new user instruction.
- The user subsequently authorized posture repair from new V1 model_500:
  task `lz_q1l_flat_v2_posture`, classes `LZQ1LFlatV2PostureCfg` / PPO in the
  shared new config file; same `LZQ1LV2UnifiedRobot` environment.
  Standing/forward 50% each (forward 0.20--0.50 m/s), base_height=-10,
  soft_dof_pos_limit=0.95 and dof_pos_limits=-2. Preserve other reward settings.
  Short run: 100 updates, save every 10, paired CPU posture evaluation every 20
  with log-only posture monitoring (the user revoked automatic stopping). Entry:
  `tools/start_lz_q1l_v2_posture_100.sh`; outputs in logs2/flat_lz_q1l_v2_posture.
  Run pointer: `media/v2_posture_20261024/latest_training.json`.
- The posture trial automatically stopped at update 20 under its original guard.
  The user then explicitly prohibited automatic quality-based training stops.
  Posture/quality checks must only record findings; they must not terminate
  training. Evaluation subprocess failures are logged and training continues.
  Do not add an early-stop policy again without an explicit user instruction.
- The user then authorized continuing overnight to 500 total posture updates.
  Resume the posture model_20 checkpoint, including both optimizer states and
  the saved PPO learning rate, for 480 additional updates. Configuration and
  rewards remain unchanged; save every 10 and evaluate every 20, log-only.
  Launcher: `tools/start_lz_q1l_v2_posture_500.sh`; use a new logs2 run directory.
- On 2026-10-25 the user authorized three gait-reward changes and initial weight
  calibration. New task `lz_q1l_flat_v2_gait` uses `LZQ1LV2GaitRobot` and
  `LZQ1LFlatV2GaitCfg` / PPO in the same new-lineage files. Keep the completed
  posture task intact. It replaces old foot_clearance with swing_clearance=-0.20
  (10 mm deficit), adds feet_slide=-0.20 and rear_force_balance=-1.0 (1 s rolling
  average, 10% imbalance deadband, 10 N minimum rear total force guard).
  All three apply only to forward commands, with command/reset history handling.
  The standing/forward mixture, PD/physics, height and joint-limit rewards remain
  inherited. The user subsequently authorized 200 additional gait updates from
  posture model_500, saving every 50 (50/100/150/200 in the new stage). Restore
  full weights, both optimizers and saved PPO learning rate; reset only the stage
  update counter. Launcher: `tools/start_lz_q1l_v2_gait_200.sh`; preserve 4096
  environments and wait in detached tmux for at least 12288 MiB free on the GPU
  with the greatest free memory. Training diagnostics are log-only, with no
  quality-based early stop. Pointer: `media/v2_gait_20261025/latest_training.json`. See
  `docs/lz_q1l_new_v2_gait_review.md` for exact formulas and calibration results.
  Validation entry: `tools/check_lz_q1l_v2_gait.py` (frozen models, no training).
  The initial gait launch exited with CUDA OOM during the first PPO update,
  completing zero updates. The queue now retries only CUDA OOM before the first
  completed update, preserving each failed attempt in a separate directory and
  increasing the free-memory threshold by 1024 MiB up to 16384 MiB. Other failures
  remain visible; do not silently discard completed training updates.

## Training log storage

- After stand/front300 evaluation showed worse walking roll, the user prioritized
  force balance and body roll, explicitly keeping source height settings.
  Authorized 200 NEW updates from upright model_150 (not stand/front300).
  Task `lz_q1l_flat_v2_balance_roll`, classes `LZQ1LFlatV2BalanceRollCfg` / PPO
  and `LZQ1LV2BalanceRollRobot`, appended in the shared new-lineage files.
  Inherit upright config directly: base_height=-30, orientation=-1,
  rear_force_balance=-1 and all other source settings remain unchanged.
  Only add front_force_balance=-1 (same 1 s window and normalized formula as
  the rear pair) and body_roll=-1 (squared roll angle in radians, all commands).
  No extra standing height or standing orientation rewards in this task.
  Restore full weights, both optimizers and saved PPO LR; 4096 environments,
  100 rollout steps, save 50/100/150/200, no quality-based early stop.
  Launcher: `tools/start_lz_q1l_v2_balance_roll_200.sh`; pointer:
  `media/v2_balance_roll_20261025/latest_training.json`; validation:
  `tools/check_lz_q1l_balance_roll.py`, `media/balance_roll_check_20261025`.

- The user authorized 300 additional updates from upright model_150 on 2026-10-25.
  Task `lz_q1l_flat_v2_stand_front` uses `LZQ1LV2StandFrontRobot`, appended in
  the same new-lineage environment/config files. Add standing-only height=-60
  and orientation=-1 extras after 0.5 s zero-command grace (combined strengths
  -90/-2); inherited movement height=-30 and orientation=-1 remain unchanged.
  Add straight-walking front_force_balance=-1 using 1 s average loads, 10%
  imbalance tolerance, 1 N denominator epsilon and 10 N minimum pair support.
  Clear histories on resets/resampling/command changes; do not constrain
  instantaneous alternating support. Preserve all other rewards and physics.
  Restore full weights, both optimizers and saved PPO learning rate; 4096 envs,
  100 rollout steps, save each 50 through 300, no quality-based early stop.
  Launcher: `tools/start_lz_q1l_v2_stand_front_300.sh`; pointer:
  `media/v2_stand_front_20261025/latest_training.json`. Validation entry:
  `tools/check_lz_q1l_stand_front.py`; results in `media/stand_front_check_20261025`.

- After checkpoint evaluation showed increasing roll, the user authorized 150
  additional upright updates from height/balance model_50 (the completed
  height_balance_from_gait200_300_20261025_133044_3370054 run). New task
  lz_q1l_flat_v2_upright uses the same LZQ1LV2GaitHeightRobot; the sole reward
  change is orientation=-1.0 instead of -0.2. Keep base_height=-30 and all other
  rewards/physics/commands unchanged. Restore both optimizers and saved PPO LR;
  stage checkpoints 50/100/150, no quality-based early stop. Launcher:
  tools/start_lz_q1l_v2_upright_150.sh; pointer:
  media/v2_upright_20261025/latest_training.json. Original height300 files remain
  archived and must not be overwritten or confused with this new stage.

- Following the completed gait200 comparison, the user requested stronger
  anti-sinking penalties and a proposal for lateral load allocation. Candidate
  `lz_q1l_flat_v2_gait_height` inherits the gait task with base_height=-30
  instead of -10; target remains 0.30 m. The user then authorized implementing
  standing load distribution. It now uses LZQ1LV2GaitHeightRobot, extending the
  gait environment with stand_pair_force_balance=-0.5 and stand_low_load=-0.2.
  Use 0.5 s force averages/grace, 10% per-pair normalized imbalance tolerance,
  and a 10% total-force minimum share per foot. Both terms are standing-only;
  clear each environment's history on resets/resampling/command changes.
  The user subsequently authorized 300 new updates from the completed gait
  model_200, saving every 50. Restore full weights, both optimizers and the saved
  PPO learning rate. New stage labels are 50/100/150/200/250/300. Launcher:
  tools/start_lz_q1l_v2_gait_height_300.sh; pointer:
  media/v2_gait_height_20261025/latest_training.json. Keep 4096 environments,
  choose greatest free GPU memory with >=12288 MiB, and monitor log-only.
  CPU rule/simulation checks and
  frozen-trace calibration passed in media/stand_balance_check_20261025; see
  `docs/lz_q1l_gait_height_balance_proposal.md`. Keep the completed gait200
  configuration and checkpoint intact for comparisons.

- The former `HIMLoco/legged_gym/logs` directory was renamed to `logs1`.
- Original V1--V8 and other historical experiments are under `HIMLoco/legged_gym/logs1`.
- The new V1 support model is under `HIMLoco/legged_gym/logs2/flat_lz_q1l_v1_support`.
- Use `runner.log_directory = 'logs2'` for the new training lineage; the shared
  task registry defaults to `logs1` for older tasks. Do not recreate `logs`.
- Historical snapshots and logs retain the paths used at execution time.
  `HIMLoco/legged_gym/logs2/path_migration.json` records the relocation.

## Simulation defaults

The user requires all future ordinary LZ-Q1L simulations and recordings to use
the established combination automatically: pure left translation uses mirrored
observations/actions; right translation and other commands use the original
controller. Do not ask the user to select a policy mode or add a mirror flag.

- Obtain the policy through `runner.get_inference_policy(device=env.device)`.
  `HIMOnPolicyRunner` delegates to `LZQ1LRobot.build_inference_policy`, which
  installs `LeftMirroredLZQ1LPolicy`. Do not wrap it again.
- This applies to `play.py`, `tools/record_lz_q1l.py`, and recording scripts
  under `media/` that use the common runner. For any newly written simulation
  or recording entrypoint, use the same factory instead of calling the actor
  directly. Keep the newest observation command synchronized when changing
  commands; preserve the real historical observations/actions.
- Global mirroring and its CLI switches have been removed at the user's
  request. Do not restore them. No policy-mode switch is needed.
- Training and checkpoint weights are unchanged. Explicit baseline or
  estimator/noise ablation tools may use the raw actor intentionally, but
  must label it as a baseline, never as the default demonstration controller.
- Keep archived experiment snapshots and previously recorded videos intact.
  Use new output directories for new runs. Prior replay plans pin source hashes
  and are historical records, not launchers for the current default.
- Continue using the existing project `.env` / Isaac Gym environment. Check GPU
  availability before runs; GPU 2 has been used when idle. Do not affect other
  users' GPU jobs.

Details and recorded limitations: `docs/lz_q1l_mirror.md`.
