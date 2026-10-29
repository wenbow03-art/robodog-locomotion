from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('media/new_lineage_audit_20261026');d=json.loads((root/'analysis.json').read_text());models=json.loads((root/'models.json').read_text());lookup={r['label']:r for r in models}
stages=['newv1','unified','posture','gait','height','upright','standfront','balanceroll']
titles=['New V1 / scratch','Unified commands / from NewV1-500','Posture repair / from NewV1-500','Gait repair / from Posture-500','Height + stand balance / from Gait-200','Upright / from Height-50','Stand extras + front / from Upright-150','Front + roll / from Upright-150']
for speed,name in [(.35,'walking'),(0.,'standing')]:
 fig,axes=plt.subplots(2,4,figsize=(15,7),sharey=True,constrained_layout=True)
 for ax,stage,title in zip(axes.flat,stages,titles):
  rows=sorted([g for g in d['groups'] if lookup[g['model']]['stage']==stage and g['speed']==speed and 'roll_abs_mean_deg' in g],key=lambda g:lookup[g['model']]['iteration'])
  x=[lookup[g['model']]['iteration'] for g in rows];y=[g['roll_abs_mean_deg']['mean'] for g in rows];err=[g['roll_abs_mean_deg']['std'] for g in rows]
  ax.errorbar(x,y,yerr=err,marker='o',capsize=3,lw=1.5);ax.set_title(title,fontsize=10);ax.set_xlabel('Stage update');ax.grid(alpha=.2);ax.set_ylim(0,32)
 axes[0,0].set_ylabel('Mean absolute roll (deg)');axes[1,0].set_ylabel('Mean absolute roll (deg)')
 fig.suptitle('%s (vx=%.2f m/s) | checkpoint sweep | 3 starts; low roll alone does not imply good walking'%(name,speed))
 fig.savefig(root/(name+'_roll.png'),dpi=150);fig.savefig(root/(name+'_roll.pdf'));plt.close(fig)
