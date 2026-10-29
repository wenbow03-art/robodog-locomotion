from pathlib import Path
import json
import numpy as np
root=Path('media/new_lineage_audit_20261026');inventory=json.loads((root/'inventory.json').read_text());result=[]
for item in inventory:
 path=Path(item['run'])/'direction_metrics.jsonl'
 if not path.exists() or not item['available']:continue
 rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
 for checkpoint in item['available']:
  if checkpoint%50:continue
  window=[x for x in rows if checkpoint-20<x['update']<=checkpoint]
  if not window:continue
  for direction in ['stand','forward']:
   selected=[x['directions'][direction] for x in window if direction in x['directions']]
   if not selected:continue
   w=np.array([x['samples'] for x in selected],dtype=float)
   metrics={k:np.average([x[k] for x in selected],weights=w,axis=0).tolist() for k in ['velocity','abs_velocity_error','force_z_N','contact','gap_m']}
   result.append(dict(stage=item['stage'],checkpoint=checkpoint,direction=direction,updates=[window[0]['update'],window[-1]['update']],samples=int(w.sum()),metrics=metrics))
(root/'training_direction_windows.json').write_text(json.dumps(result,indent=2)+'\n')
for r in result:
 if r['stage']=='gait':
  m=r['metrics'];print(r['checkpoint'],r['direction'],'Fz',[round(x,1) for x in m['force_z_N']],'contact',[round(x*100,1) for x in m['contact']])
