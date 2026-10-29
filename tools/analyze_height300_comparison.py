"""Compare exact videos and paired gait200/height300 trials; preserve all failures."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'media/height300_analysis_20261025';out.mkdir(exist_ok=True)
    paired=ROOT/'media/height300_comparison_20261025/report.json'
    report=json.loads(paired.read_text());groups=[];video=[];sources=[paired]
    labels=['gait200','height300']
    for model in labels:
        for speed in (0.,.2,.35,.5):
            trials=[r for r in report['trials'] if r['model']==model and r['command']==speed]
            valid=[r for r in trials if 'vx' in r]
            row=dict(model=model,speed=speed,trials=len(trials),failures=sum(r['failed'] for r in trials),
                     valid_samples=[r['valid_samples'] for r in trials])
            if valid:
                for k in valid[0]:
                    if k in ('model','command','start','failed','valid_samples'):continue
                    a=np.array([r[k] for r in valid],dtype=float)
                    assert np.isfinite(a).all(),(model,speed,k)
                    row[k]=dict(mean=a.mean(0).tolist(),std=a.std(0).tolist())
            groups.append(row)
    folders=[ROOT/'media/v2_gait200_20261025/run_20261025_130141_vBlU/result',
             ROOT/'media/v2_height300_20261025/run_20261025_164436_aZU7/result']
    for model,folder in zip(labels,folders):
        meta=json.loads((folder/'metadata.json').read_text());trace=json.loads((folder/'trace.json').read_text())
        assert not meta['resets']
        assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['checkpoint_sha256']
        sources.extend([folder/'metadata.json',folder/'trace.json'])
        for segment in (0,1):
            t=[r for r in trace if r['segment']==segment][50:];f=np.array([r['contact_force_z'] for r in t]).clip(0)
            g=np.array([r['gap_mm'] for r in t]);v=np.array([r['velocity'] for r in t]);h=np.array([r['position'][2] for r in t]);force=f.mean(0)
            row=dict(model=model,segment=segment,height_cm=float(h.mean()*100),height_mae_cm=float(np.abs(h-.3).mean()*100),
                height_std_mm=float(h.std()*1000),force_n=force.tolist(),load_share_pct=(force/force.sum()*100).tolist(),
                rear_force_difference_n=float(abs(force[0]-force[1])),front_force_difference_n=float(abs(force[2]-force[3])),
                contact_pct=((f>1).mean(0)*100).tolist(),gap_max_mm=g.max(0).tolist(),
                vx=float(v[:,0].mean()),vx_mae=float(np.abs(v[:,0]-(.35 if segment else 0)).mean()),vy_mae=float(np.abs(v[:,1]).mean()))
            video.append(row)
    data=dict(video=video,paired=groups,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol='Videos: last 7 seconds, 350 samples per segment. Paired: stand/.2/.35/.5 m/s x 3 same initial states per policy, 2s standing preroll, 8s command, stats command seconds 1-8 before first failure.',
        limitations='One trained model per variant, three initial states, flat ground. Cross-start std does not measure independent training seed uncertainty. Contact slip is sphere-bottom rigid velocity estimate. Changes in the two objectives and added training are not causally isolated.')
    (out/'analysis.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    def get(model,speed):return next(g for g in groups if g['model']==model and g['speed']==speed)
    fig,axes=plt.subplots(2,3,figsize=(17,8),constrained_layout=True)
    for ax,speed,key,title,unit in [(axes[0,0],0.,'force_n','Standing support force','N'),
            (axes[0,1],.35,'force_n','Walking support force (0.35 m/s)','N'),
            (axes[1,0],.35,'gap_max_mm','Walking peak clearance (0.35 m/s)','mm')]:
        x=np.arange(4)
        for i,(model,color) in enumerate(zip(labels,['#718096','#168c89'])):
            a=get(model,speed)[key]
            ax.bar(x+(i-.5)*.35,a['mean'],yerr=a['std'],width=.35,label=model,color=color,capsize=3)
        ax.set_xticks(x);ax.set_xticklabels(['LH','RH','LF','RF']);ax.set_title(title);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];x=np.arange(4)
    for i,(model,color) in enumerate(zip(labels,['#718096','#168c89'])):
        a=[get(model,s)['height_cm'] for s in (0.,.2,.35,.5)]
        ax.bar(x+(i-.5)*.35,[r['mean'] for r in a],yerr=[r['std'] for r in a],width=.35,label=model,color=color,capsize=3)
    ax.axhline(30,color='red',ls='--',label='30 cm target');ax.set_xticks(x);ax.set_xticklabels(['Stand','0.20','0.35','0.50'])
    ax.set_title('Body height');ax.set_ylabel('cm');ax.set_ylim(25,35);ax.legend();ax.grid(axis='y',alpha=.2)
    for ax,key,title,unit in [(axes[0,2],'roll_mean_deg','Mean body roll','deg'),
                              (axes[1,2],'height_mae_cm','Absolute height error vs 30 cm','cm')]:
        for i,(model,color) in enumerate(zip(labels,['#718096','#168c89'])):
            a=[get(model,s)[key] for s in (0.,.2,.35,.5)]
            ax.bar(x+(i-.5)*.35,[r['mean'] for r in a],yerr=[r['std'] for r in a],width=.35,label=model,color=color,capsize=3)
        ax.set_xticks(x);ax.set_xticklabels(['Stand','0.20','0.35','0.50']);ax.set_title(title);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend();fig.suptitle('Gait200 vs height/balance300 | three paired starts | mean +/- SD')
    fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.pdf');plt.close(fig)
    lines=['# 高度与承重300轮 vs 步态200轮','',data['protocol'],'',
        '## 视频原始数据','',
        '|模型/动作|高度cm|高度误差cm|左后/右后N|左前/右前N|后腿力差N|前腿力差N|速度绝对误差m/s|',
        '|---|---:|---:|---|---|---:|---:|---:|']
    for r in video:
        f=r['force_n'];lines.append('|%s/%s|%.3f|%.3f|%.2f/%.2f|%.2f/%.2f|%.2f|%.2f|%.5f|'%(r['model'],['站立','直行'][r['segment']],r['height_cm'],r['height_mae_cm'],*f,r['rear_force_difference_n'],r['front_force_difference_n'],r['vx_mae']))
    lines+=['','## 三个初始状态配对复测均值','',
        '|模型|指令m/s|失败/3|高度cm|高度误差cm|后腿力差N|前腿力差N|速度绝对误差m/s|右后滑动m/s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in groups:
        lines.append('|%s|%.2f|%d|%.3f|%.3f|%.2f|%.2f|%.5f|%.5f|'%(r['model'],r['speed'],r['failures'],r['height_cm']['mean'],r['height_mae_cm']['mean'],r['rear_force_difference_n']['mean'],r['front_force_difference_n']['mean'],r['vx_mae']['mean'],r['contact_slip_m_s']['mean'][1]))
    lines+=['',data['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    for r in groups:
        print('DETAIL',r['model'],r['speed'],json.dumps({k:r[k] for k in ('force_n','contact_pct','roll_mean_deg','pitch_mean_deg','height_std_mm','standing_penalties','vy_mae','yaw_mae','gap_max_mm')}))


if __name__=='__main__':main()
