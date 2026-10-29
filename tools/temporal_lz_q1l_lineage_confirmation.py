from pathlib import Path
import json
import numpy as np
root=Path('media/new_lineage_audit_20261026/confirm');result=[]
for folder in sorted(root.iterdir()):
 if not (folder/'eval/report.json').exists():continue
 z=np.load(folder/'eval/traces.npz');arr={k:z[k] for k in ['fz','quat','valid','velocity','height']}
 for ci,speed in enumerate([0.,.2,.35,.5]):
  for start,end in [(150,500),(500,1100)]:
   rows=[]
   for si in range(3):
    idx=ci*3+si;mask=arr['valid'][start:end,idx];a={k:v[start:end,idx][mask] for k,v in arr.items()};f=a['fz'].clip(0).mean(0);x,y,z,w=a['quat'].T;roll=np.rad2deg(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)))
    rows.append(dict(roll=float(abs(roll).mean()),rear=float(abs(f[0]-f[1])),front=float(abs(f[2]-f[3])),left_share=float((f[0]+f[2])/f.sum()*100),height=float(a['height'].mean()*100),vx_mae=float(abs(a['velocity'][:,0]-speed).mean())))
   result.append(dict(model=folder.name,speed=speed,seconds_after_command=[start*.02-2,end*.02-2],mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0]},per_start=rows))
(root/'temporal.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for r in result:
 if r['speed']==0:print(r['model'],r['seconds_after_command'],r['mean'])
