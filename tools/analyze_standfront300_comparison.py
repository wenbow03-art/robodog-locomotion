"""Summarize paired frozen policies; SD is across starts, not training seeds."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'media/standfront300_comparison_20261025'
data=json.loads((out/'eval/report.json').read_text())
manifest=json.loads((out/'eval/manifest.json').read_text())
metrics=['roll_mean_deg','roll_abs_mean_deg','pitch_abs_mean_deg','height_cm','height_mae_cm','height_std_mm','vx','vx_mae','vy_mae','yaw_mae','force_n','rear_force_difference_n','front_force_difference_n','contact_pct','gap_max_mm','gap_p95_mm','contact_slip_m_s','loaded_slip_m_s','roll_std_deg','pitch_std_deg','front_balance_raw','stand_extra_raw']
groups=[]
for model in manifest['labels']:
 for speed in [0.,.2,.35,.5]:
  trials=[r for r in data['trials'] if r['model']==model and r['command']==speed]
  row=dict(model=model,speed=speed,failures=sum(t['failed'] for t in trials),n=len(trials))
  for key in metrics:
   a=np.array([t[key] for t in trials if key in t],dtype=float)
   row[key]=dict(mean=a.mean(0).tolist(),std=a.std(0).tolist())
  groups.append(row)
comparisons=[]
for speed in [0.,.2,.35,.5]:
 new=next(g for g in groups if g['model']=='standfront300' and g['speed']==speed)
 for baseline in ['upright150']:
  old=next(g for g in groups if g['model']==baseline and g['speed']==speed)
  comparisons.append(dict(speed=speed,baseline=baseline,relative_change_pct={k:100*(new[k]['mean']-old[k]['mean'])/abs(old[k]['mean']) if abs(old[k]['mean'])>1e-8 else None for k in metrics if isinstance(new[k]['mean'],float)}))
result=dict(groups=groups,comparisons=comparisons,resets=data['resets'],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'eval/report.json',out/'eval/manifest.json',Path(__file__)]},limitations='3 paired initial states, one terrain, one trained policy per condition; SD across starts is not training-seed uncertainty. Different joint-start protocol from historical reports; compare only within this batch. Foot contact Fz > 1 N, loaded Fz > 10 N. Gap maxima do not measure gait symmetry alone. Statistics command seconds 1-8 after 2s standing preroll. No learning.')
(out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
lines=['# Standfront300 checkpoint comparison','',result['limitations'],'','|Model|Command m/s|Abs roll deg|Abs pitch deg|Height cm|Height MAE cm|Vx MAE m/s|Rear force gap N|Front force gap N|Failures|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in groups:
 lines.append('|%s|%.2f|%.2f ± %.2f|%.2f|%.2f|%.2f|%.4f|%.2f|%.2f|%d/3|'%(g['model'],g['speed'],g['roll_abs_mean_deg']['mean'],g['roll_abs_mean_deg']['std'],*[g[k]['mean'] for k in ['pitch_abs_mean_deg','height_cm','height_mae_cm','vx_mae','rear_force_difference_n','front_force_difference_n']],g['failures']))
(out/'report.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for ax,key,title,unit in [(axes[0,0],'roll_abs_mean_deg','Body roll magnitude','deg'),(axes[0,1],'height_cm','Body height','cm'),(axes[1,0],'rear_force_difference_n','Rear pair mean-force difference','N'),(axes[1,1],'front_force_difference_n','Front pair mean-force difference','N')]:
 for model in manifest['labels']:
  gs=[g for g in groups if g['model']==model]
  ax.errorbar([g['speed'] for g in gs],[g[key]['mean'] for g in gs],yerr=[g[key]['std'] for g in gs],marker='o',capsize=3,label=model)
 ax.set_title(title);ax.set_xlabel('Forward command (m/s; 0 = stand)');ax.set_ylabel(unit);ax.grid(alpha=.2)
axes[0,0].legend();axes[0,1].axhline(30,color='gray',ls='--')
fig.suptitle('Frozen policy comparison | 3 paired starts, mean +/- SD')
fig.savefig(out/'comparison.png',dpi=150);fig.savefig(out/'comparison.pdf');plt.close(fig)
print('\n'.join(lines))
for g in groups:
 if g['speed'] in [0.,.35] and g['model'] in ['upright150','standfront300']:print('DETAIL',json.dumps(g))
