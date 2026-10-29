"""Archive final within-run scalar windows; do not compare total reward across changed rewards."""
from pathlib import Path
import json,numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out=Path('media/new_lineage_audit_20261026');inv=json.loads((out/'inventory.json').read_text());result=[]
for row in inv:
 folder=Path(row['run']);events=list(folder.glob('events.out.tfevents.*'))
 if not events or not row['available']:continue
 ea=EventAccumulator(str(folder),size_guidance={'scalars':0});ea.Reload();tags=ea.Tags()['scalars'];records={}
 wanted=[t for t in tags if t.startswith(('Train/','Episode/rew_','Loss/learning_rate'))]
 for tag in wanted:
  v=ea.Scalars(tag);v=sorted({x.step:x.value for x in v}.items());n=min(50,len(v)//2)
  if not n:continue
  recent=np.array([x[1] for x in v[-n:]]);prior=np.array([x[1] for x in v[-2*n:-n]])
  records[tag]=dict(n=n,previous_mean=float(prior.mean()),last_mean=float(recent.mean()),last_std=float(recent.std()),last_step=v[-1][0])
 result.append(dict(stage=row['stage'],run=row['run'],metrics=records))
(out/'training_windows.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for r in result:
 print(r['stage'],{k:v for k,v in r['metrics'].items() if k in ['Train/mean_reward','Train/mean_episode_length','Train/mean_noise_std','Episode/rew_orientation','Episode/rew_body_roll','Loss/learning_rate']})
