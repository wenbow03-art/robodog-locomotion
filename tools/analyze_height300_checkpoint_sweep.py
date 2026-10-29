"""Locate roll degradation across frozen checkpoints, with training context."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'media/height300_checkpoint_sweep_20261025'
    raw=json.loads((folder/'report.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    groups=[]
    for update in manifest['additional_updates']:
        for speed in (0.,.2,.35,.5):
            trials=[r for r in raw['trials'] if r['additional_update']==update and r['command']==speed]
            valid=[r for r in trials if 'vx' in r]
            row=dict(update=update,speed=speed,trials=len(trials),failures=sum(t['failed'] for t in trials),
                     valid_samples=[t['valid_samples'] for t in trials])
            for k in ('roll_mean_deg','pitch_mean_deg','height_cm','height_mae_cm','vx_mae',
                      'rear_force_difference_n','front_force_difference_n','standing_penalties','force_n','contact_pct'):
                if valid:
                    v=np.array([t[k] for t in valid]);row[k]=dict(mean=v.mean(0).tolist(),std=v.std(0).tolist())
            groups.append(row)
    run=Path(manifest['checkpoints'][-1]).parent
    ea=EventAccumulator(str(run),size_guidance={'scalars':0});ea.Reload()
    tags=['Episode/rew_orientation','Episode/rew_base_height','Episode/rew_stand_pair_force_balance','Train/mean_reward']
    series={tag:[dict(update=e.step+1,value=e.value) for e in ea.Scalars(tag)] for tag in tags}
    assert all(len(a)==300 for a in series.values())
    blocks=[]
    for lo in range(0,300,25):
        blocks.append(dict(start=lo+1,end=lo+25,means={tag:float(np.mean([r['value'] for r in a[lo:lo+25]])) for tag,a in series.items()}))
    sources=[folder/'report.json',folder/'manifest.json']+list(run.glob('events.out.tfevents.*'))
    result=dict(groups=groups,training_blocks=blocks,training_series=series,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        limitations='Checkpoints every 50 additional updates only; cannot identify exact first bad update. Three paired initial states and one terrain. Training orientation cost includes both roll and pitch and randomized/exploratory trajectories; not a roll-angle measurement.')
    (folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for speed,label in [(0.,'Stand'),(.2,'0.20 m/s'),(.35,'0.35 m/s'),(.5,'0.50 m/s')]:
        a=[g for g in groups if g['speed']==speed];x=[g['update'] for g in a]
        for ax,key,title,unit in [(axes[0,0],'roll_mean_deg','Measured body roll','deg'),
                                  (axes[0,1],'height_cm','Body height','cm'),
                                  (axes[1,0],'front_force_difference_n','Front pair mean-force difference','N')]:
            ax.errorbar(x,[g[key]['mean'] for g in a],yerr=[g[key]['std'] for g in a],marker='o',capsize=3,label=label)
            ax.set_title(title);ax.set_ylabel(unit);ax.set_xlabel('Additional updates (0 = source gait200)');ax.grid(alpha=.2)
    axes[0,1].axhline(30,color='gray',ls='--');axes[0,0].legend()
    a=series['Episode/rew_orientation'];x=np.array([r['update'] for r in a]);y=-np.array([r['value'] for r in a])
    axes[1,1].plot(x,y,alpha=.3);axes[1,1].plot(x[9:],np.convolve(y,np.ones(10)/10,mode='valid'))
    axes[1,1].set_title('Training orientation penalty magnitude (roll + pitch)');axes[1,1].set_xlabel('Additional updates');axes[1,1].grid(alpha=.2)
    fig.suptitle('Height/balance training: checkpoint sweep | 3 paired starts, mean +/- SD')
    fig.savefig(folder/'checkpoint_trends.png',dpi=160);fig.savefig(folder/'checkpoint_trends.pdf');plt.close(fig)
    lines=['# 本次300轮：侧倾何时加重','',
        '以下轮数为从gait200继续训练后的新增轮数；0代表续训前模型。每个检查点测试站立和3种直行速度，每种3个相同初始状态。',
        '', '|新增轮数|站立侧倾°|0.20m/s侧倾°|0.35m/s侧倾°|0.50m/s侧倾°|失败/12|',
        '|---|---:|---:|---:|---:|---:|']
    for update in manifest['additional_updates']:
        a=[g for g in groups if g['update']==update]
        lines.append('|%d|%.2f|%.2f|%.2f|%.2f|%d|'%(update,*[r['roll_mean_deg']['mean'] for r in a],sum(r['failures'] for r in a)))
    lines+=['',result['limitations']]
    (folder/'analysis.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    for g in groups:
        if g['speed'] in (0.,.35):print('DETAIL',json.dumps(g))


if __name__=='__main__':main()
