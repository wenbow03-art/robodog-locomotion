"""Aggregate measured behavior, with no cross-reward total-score comparison."""
import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory
report=json.loads((out/'eval/report.json').read_text());manifest=json.loads((out/'eval/manifest.json').read_text())
archive=np.load(out/'eval/traces.npz');arr={k:archive[k] for k in ['valid','dof_pos','joint_target','fz','gap']};names=manifest['dof_names'];groups=[]
for mi,label in enumerate(manifest['labels']):
 for ci,speed in enumerate([0.,.2,.35,.5]):
  trials=[r for r in report['trials'] if r['model']==label and r['command']==speed]
  group=dict(model=label,speed=speed,failures=sum(t['failed'] for t in trials))
  keys=[k for k in trials[0] if k not in ['checkpoint_iteration','model','command','start','failed','valid_samples']]
  for k in keys:
   vals=np.asarray([t[k] for t in trials if k in t],dtype=float)
   group[k]=dict(mean=np.nanmean(vals,axis=0).tolist(),std=np.nanstd(vals,axis=0).tolist(),finite_count=np.isfinite(vals).sum(0).tolist())
  vals=[]
  for si in range(3):
   idx=mi*12+ci*3+si;valid=arr['valid'][150:,idx];d={k:arr[k][150:,idx][valid] for k in ['dof_pos','joint_target','fz','gap']}
   if not valid.any():continue
   f=d['fz'].clip(0);meanf=f.mean(0)
   vals.append(dict(actual_joint_mean_deg=np.rad2deg(d['dof_pos']).mean(0),target_joint_mean_deg=np.rad2deg(d['joint_target']).mean(0),target_tracking_mae_deg=np.rad2deg(abs(d['dof_pos']-d['joint_target'])).mean(0),left_share_pct=100*(meanf[0]+meanf[2])/meanf.sum(),all_four_contact_pct=100*(f>1).all(1).mean(),air_above5mm_pct=100*((d['gap']>.005)&(f<=1)).mean(0)))
  for k in vals[0] if vals else []:
   v=np.asarray([r[k] for r in vals]);group[k]=dict(mean=v.mean(0).tolist(),std=v.std(0).tolist())
  groups.append(group)
result=dict(groups=groups,dof_names=names,feet_names=manifest['feet_names'],resets=report['resets'],protocol=manifest['protocol'])
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
result=clean(result)
(out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
keys=['roll_abs_mean_deg','pitch_abs_mean_deg','rear_force_difference_n','front_force_difference_n','left_share_pct','height_cm','vx_mae','all_four_contact_pct']
with (out/'metrics.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','speed','failures']+keys+['RH_max_gap_mm','RH_contact_slip_m_s']);
 for g in groups:
  w.writerow([g['model'],g['speed'],g['failures']]+[g[k]['mean'] if k in g else None for k in keys]+[g.get('gap_max_mm',{'mean':[None,None]})['mean'][1],g.get('contact_slip_m_s',{'mean':[None,None]})['mean'][1]])
for g in groups:
 if g['speed']==.35 and 'roll_abs_mean_deg' in g:
  print('%-16s roll=%5.2f rear=%5.1f front=%5.1f left=%5.1f%% h=%5.2f vxerr=%.3f RHlift=%5.1f slip=%.3f fail=%d'%(g['model'],*[g[k]['mean'] for k in keys[:1]+keys[2:7]],g['gap_max_mm']['mean'][1],g['contact_slip_m_s']['mean'][1],g['failures']))
print('RESETS',len(report['resets']));print('JOINT_NAMES',names)
