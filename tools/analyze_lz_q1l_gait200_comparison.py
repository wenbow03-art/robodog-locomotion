"""Analyze exact demonstration traces and independent paired simulation checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'media/gait200_analysis_20261025'
VIDEOS=[ROOT/'media/v2_posture500_20261025/run_20261025_082315_antS/result',
        ROOT/'media/v2_gait200_20261025/run_20261025_130141_vBlU/result']
LABELS=['posture500','gait200']


def main():
    OUT.mkdir(exist_ok=True)
    rows=[];traces=[];sources={}
    for label,folder in zip(LABELS,VIDEOS):
        meta=json.loads((folder/'metadata.json').read_text());trace=json.loads((folder/'trace.json').read_text())
        assert not meta['resets']
        assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['checkpoint_sha256']
        for name in ('metadata.json','trace.json','config.json'):
            p=folder/name;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        traces.append(trace)
        for segment in (0,1):
            a=[r for r in trace if r['segment']==segment][50:]
            f=np.array([r['contact_force_z'] for r in a]).clip(0);g=np.array([r['gap_mm'] for r in a]);v=np.array([r['velocity'] for r in a]);h=np.array([r['position'][2] for r in a])
            peaks=[];maxair=[];counts=[]
            for j in range(4):
                air=f[:,j]<=1;changes=np.diff(np.r_[False,air,False].astype(int));starts=np.where(changes==1)[0];ends=np.where(changes==-1)[0]
                complete=[(s,e) for s,e in zip(starts,ends) if s>0 and e<len(a)]
                peaks.append(float(np.mean([g[s:e,j].max() for s,e in complete])) if complete else None)
                maxair.append(float(max(ends-starts)*.02) if len(starts) else 0.)
                counts.append(len(complete))
            forces=f.mean(0)
            rows.append(dict(model=label,segment=segment,contact_pct=((f>1).mean(0)*100).tolist(),
                gap_max_mm=g.max(0).tolist(),gap_p95_mm=np.percentile(g,95,axis=0).tolist(),
                mean_swing_peak_mm=peaks,full_air_intervals=counts,max_noncontact_s=maxair,
                force_n=forces.tolist(),rear_force_difference_n=float(abs(forces[0]-forces[1])),
                rear_imbalance=float(abs(forces[0]-forces[1])/(forces[0]+forces[1]+1)),
                height_cm=float(h.mean()*100),height_std_mm=float(h.std()*1000),
                vx=float(v[:,0].mean()),vx_mae=float(np.abs(v[:,0]-(.35 if segment else 0)).mean()),
                vy_mae=float(np.abs(v[:,1]).mean())))
    paired_path=ROOT/'media/gait200_comparison_20261025/report.json'
    paired=json.loads(paired_path.read_text());groups=[]
    sources[str(paired_path)]=hashlib.sha256(paired_path.read_bytes()).hexdigest()
    for label in LABELS:
        for speed in (0.,.2,.35,.5):
            samples=[t for t in paired['trials'] if t['model']==label and t['command']==speed]
            valid=[t for t in samples if 'vx' in t]
            row=dict(model=label,speed=speed,failures=sum(t['failed'] for t in samples),trials=len(samples),
                     valid_samples=[t['valid_samples'] for t in samples])
            for k in ('vx','vx_mae','height_cm','height_std_mm','contact_pct','gap_max_mm','gap_p95_mm','force_n',
                      'rear_force_difference_n','rear_imbalance','contact_slip_m_s','loaded_slip_m_s','raw_penalties'):
                if valid:
                    a=np.array([t[k] for t in valid],dtype=float)
                    assert np.isfinite(a).all(),(label,speed,k)
                    row[k]=dict(mean=a.mean(0).tolist(),std=a.std(0).tolist())
            groups.append(row)
    report=dict(video_cases=rows,paired_groups=groups,source_sha256=sources,
        protocol='Video stats: last 7 seconds of each 8-second segment, 350 samples at 50 Hz. Paired stats: three initial states, command seconds 1-8, pre-failure samples only. Std across initial states, not independent trained seeds.',
        limitations='No causal isolation between the three rewards or added training. Maximum clearance is not average step height. Force imbalance uses long-window averages, not instantaneous alternating support. Same flat terrain only.')
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    feet=['LH','RH','LF','RF'];x=np.arange(4);colors=['#718096','#167b8c']
    walking=[r for r in rows if r['segment']==1]
    for ax,key,title,ylabel in zip(axes.flat,['gap_max_mm','force_n','contact_pct','mean_swing_peak_mm'],
        ['Peak foot clearance','Mean vertical support force','Contact fraction during walking','Mean peak per complete swing'],['mm','N','%','mm']):
        for i,(row,color) in enumerate(zip(walking,colors)):
            ax.bar(x+(i-.5)*.35,row[key],width=.35,color=color,label=LABELS[i])
        ax.set_xticks(x);ax.set_xticklabels(feet);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend();fig.suptitle('Exact video traces | command 0.35 m/s | last 7 seconds')
    fig.savefig(OUT/'walking_comparison.png',dpi=160);plt.close(fig)
    text=['# 步态修正200轮与原姿态500轮对比','',report['protocol'],'',
          '旧：posture500；新：gait200。脚顺序为左后、右后、左前、右前。','',
          '## 视频原始轨迹：直行0.35 m/s','',
          '|指标|旧模型|新模型|','|---|---:|---:|']
    for title,key,fmt in [('右后最大离地/mm','gap_max_mm',1),('右后平均每次摆腿峰值/mm','mean_swing_peak_mm',1),
                           ('右后平均承重/N','force_n',1),('后腿平均力差/N','rear_force_difference_n',None),
                           ('机身高度/cm','height_cm',None),('平均前进速度/m/s','vx',None),('前进速度绝对误差/m/s','vx_mae',None)]:
        values=[r[key] if fmt is None else r[key][fmt] for r in walking]
        text.append('|%s|%.4f|%.4f|'%(title,*values))
    text+=['','## 配对复测','',
           '|模型|指令m/s|失败/3|平均速度|高度cm|后腿平均力差N|右后最大离地mm（3次均值）|右后触地滑动m/s|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in groups:
        text.append('|%s|%.2f|%d|%.4f|%.2f|%.2f|%.2f|%.4f|'%(row['model'],row['speed'],row['failures'],
            row['vx']['mean'],row['height_cm']['mean'],row['rear_force_difference_n']['mean'],
            row['gap_max_mm']['mean'][1],row['contact_slip_m_s']['mean'][1]))
    text+=['','站立触地并不等于承重均匀；必须单独报告站立力分布和高度。',
           '统计为三个初始状态的配对测试，不能推断所有地形稳定性，也不能把改进单独归因于某一个奖励项。',
           '滑动为脚部碰撞球底点刚体速度估计。完整逐脚数据及跨初始状态标准差见 analysis.json。']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
