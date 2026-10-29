import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'media/roll_joint_response_20261026';d=out/'eval'
m=json.loads((d/'manifest.json').read_text());status=json.loads((d/'status.json').read_text());a=dict(np.load(d/'traces.npz'))
assert status['finite'] and a['q'].shape==(600,36,12)
names=m['dof_names'];legs=['b_l','b_r','f_l','f_r'];kinds=['pitch01','pitch02']
slots=np.array([[names.index(k+'_leg_'+leg+'_joint') for k in kinds] for leg in legs])
q=np.degrees(a['q']);target=np.degrees(a['target']);preq=np.degrees(a['pre_q'])
def roll_of(quat):
 x,y,z,w=np.moveaxis(quat,-1,0);return np.degrees(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)))
roll=roll_of(a['quat']);preroll=roll_of(a['pre_quat']);t=(np.arange(600)+1)*.02
f=a['fz'].clip(0)
rows=[]
for c in m['cases']:
 if c['impulse']!=0:continue
 i=c['index'];sl=slice(150,600)
 actual=q[sl,i][:,slots];desired=target[sl,i][:,slots];force=f[sl,i]
 row=dict(**c,roll_deg=float(roll[sl,i].mean()),force_n=force.mean(0).tolist(),
  actual_angle_deg=actual.mean(0).tolist(),target_angle_deg=desired.mean(0).tolist(),
  tracking_mae_deg=np.abs(actual-desired).mean(0).tolist(),
  actual_right_minus_left_rear_deg=(actual[:,1]-actual[:,0]).mean(0).tolist(),
  target_right_minus_left_rear_deg=(desired[:,1]-desired[:,0]).mean(0).tolist(),
  initial_roll_deg=float(preroll[0,i]),initial_q_deg=preq[0,i][slots].tolist(),initial_target_deg=target[0,i][slots].tolist(),
  initial_target_right_minus_left_rear_deg=(target[0,i][slots[1]]-target[0,i][slots[0]]).tolist())
 row['stance_angles_deg']=[actual[:,j][force[:,j]>10].mean(0).tolist() if (force[:,j]>10).any() else None for j in range(4)]
 rows.append(row)
responses=[]
for c in m['cases']:
 if c['impulse']==0:continue
 i=c['index'];j=next(z['index'] for z in m['cases'] if z['model']==c['model'] and z['speed']==c['speed'] and z['start']==c['start'] and z['impulse']==0)
 pre=slice(150,200)
 rdelta=roll[:,i]-roll[:,j];rdelta-=rdelta[pre].mean()
 delta=target[:,i][:,slots]-target[:,j][:,slots];delta-=delta[pre].mean(0)
 actdelta=q[:,i][:,slots]-q[:,j][:,slots];actdelta-=actdelta[pre].mean(0)
 rear=delta[:,1]-delta[:,0]
 responses.append(dict(**c,pre_roll_mismatch_deg=float(np.abs(roll[pre,i]-roll[pre,j]).max()),
  pre_target_mismatch_mae_deg=float(np.abs(target[pre,i]-target[pre,j]).mean()),
  delta_target_first_step_deg=delta[200].tolist(),delta_rear_target_first_step_deg=rear[200].tolist(),
  delta_rear_target_first_100ms_deg=rear[200:205].mean(0).tolist(),
  delta_roll_peak_1s_deg=float(rdelta[200:250][np.argmax(np.abs(rdelta[200:250]))]),
  delta_roll_abs_last_2s_deg=float(np.abs(rdelta[500:600]).mean()),
  delta_actual_rear_first_100ms_deg=(actdelta[200:205,1]-actdelta[200:205,0]).mean(0).tolist()))
result=dict(unperturbed=rows,responses=responses,dof_order=names,angle_order=['thigh pitch01','knee pitch02'],feet_order=legs,resets=status['resets'],
 note='Angles in signed URDF joint coordinates, not visual included knee angle. Averages over walking do not compare matched gait phase. Target/actual tracking difference includes load-dependent PD deflection. Impulse response is relative to paired sham after subtracting pre-event mean; timing alone does not prove causal direction.',
 source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [d/'manifest.json',d/'traces.npz',Path(__file__)]})
(out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for model in ['upright150','balanceroll200']:
 for speed in [0.,.35]:
  rs=[r for r in rows if r['model']==model and r['speed']==speed]
  print('BASE',model,speed,json.dumps({k:np.mean([r[k] for r in rs],axis=0).tolist() for k in ['roll_deg','force_n','actual_angle_deg','target_angle_deg','tracking_mae_deg','actual_right_minus_left_rear_deg','target_right_minus_left_rear_deg']}))
  r=next(r for r in rs if r['start']==0);print('INITIAL',model,speed,json.dumps({k:r[k] for k in ['initial_roll_deg','initial_q_deg','initial_target_deg','initial_target_right_minus_left_rear_deg']}))
  for sign in [-.5,.5]:
   rs=[r for r in responses if r['model']==model and r['speed']==speed and r['impulse']==sign]
   print('RESPONSE',model,speed,sign,json.dumps({k:np.mean([r[k] for r in rs],axis=0).tolist() for k in ['pre_roll_mismatch_deg','pre_target_mismatch_mae_deg','delta_rear_target_first_step_deg','delta_rear_target_first_100ms_deg','delta_roll_peak_1s_deg','delta_roll_abs_last_2s_deg']}))
# Show unperturbed current model, with synchronized time axes.
fig,axes=plt.subplots(4,2,figsize=(13,10),sharex=True,constrained_layout=True)
for col,speed in enumerate([0.,.35]):
 i=next(c['index'] for c in m['cases'] if c['model']=='balanceroll200' and c['speed']==speed and c['start']==0 and c['impulse']==0)
 axes[0,col].plot(t,roll[:,i]);axes[0,col].set_title('Stand' if not speed else 'Forward 0.35 m/s');axes[0,col].set_ylabel('Roll (deg)')
 for k,kind in enumerate(['Thigh','Knee']):
  for foot,label,color in [(0,'LH','tab:blue'),(1,'RH','tab:orange')]:
   axes[k+1,col].plot(t,q[:,i,slots[foot,k]],color=color,label=label+' actual')
   axes[k+1,col].plot(t,target[:,i,slots[foot,k]],color=color,ls='--',alpha=.7,label=label+' target')
  axes[k+1,col].set_ylabel(kind+' angle (deg)')
 for foot,label in enumerate(['LH','RH','LF','RF']):
  smooth=np.convolve(f[:,i,foot],np.ones(10)/10,mode='valid')
  axes[3,col].plot(t[9:],smooth,label=label)
 axes[3,col].set_ylabel('Fz (N), 0.2s mean');axes[3,col].set_xlabel('Time (s; forward starts at 2s)')
 for ax in axes[:,col]:ax.grid(alpha=.2);ax.axvline(2,color='gray',ls=':')
axes[1,0].legend(ncol=2);axes[3,0].legend(ncol=4)
fig.suptitle('Current balance/roll200: joint targets, actual angles, force and roll | unperturbed')
fig.savefig(out/'joint_roll_trace.png',dpi=150);fig.savefig(out/'joint_roll_trace.pdf');plt.close(fig)
