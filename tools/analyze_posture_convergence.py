"""Read-only analysis of the completed posture 1-500 run and periodic CPU checks."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[1]
run=Path(json.loads((ROOT/'media/v2_posture_20261024/latest_training.json').read_text())['run_dir'])
parent=Path(json.loads((run/'run_metadata.json').read_text())['source_checkpoint']).parent
out=ROOT/'media/posture_convergence_20261025'
out.mkdir(exist_ok=False)
tags=['Train/mean_reward','Train/mean_episode_length','Episode/rew_tracking_lin_vel',
      'Episode/rew_base_height','Episode/rew_stand_unsupported','Episode/rew_foot_clearance',
      'Loss/value_function','Loss/Estimation Loss','Loss/Swap Loss','Loss/surrogate',
      'Policy/mean_noise_std','Directions/forward/abs_velocity_error_0',
      'Directions/forward/contact_slip_m_s_1']
series={tag:{} for tag in tags};sources=[]
for folder in (parent,run):
    events=list(folder.glob('events.out.tfevents.*'));assert events
    sources+=events
    ea=EventAccumulator(str(folder),size_guidance={'scalars':0});ea.Reload()
    for tag in tags:
        for item in ea.Scalars(tag):
            step=item.step+(0 if tag.startswith('Directions/') else 1)
            assert step not in series[tag],(tag,step)
            assert np.isfinite(item.value)
            series[tag][step]=item.value
assert set(series['Train/mean_reward'])==set(range(1,501))
windows=[]
for lo,hi in [(51,100),(101,200),(201,300),(301,400),(401,500),(301,350),(351,400),(401,450),(451,500)]:
    metrics={}
    for tag,data in series.items():
        rows=np.array([(i,data[i]) for i in range(lo,hi+1)])
        metrics[tag]=dict(mean=float(rows[:,1].mean()),std_across_updates=float(rows[:,1].std()),
            slope_per100_updates=float(np.polyfit(rows[:,0],rows[:,1],1)[0]*100))
    windows.append(dict(start=lo,end=hi,metrics=metrics))
evaluations=[]
for path in sorted(run.glob('posture_check_*.json')):
    sources.append(path);d=json.loads(path.read_text())
    entry=dict(update=d['update'],evaluation_ok=d['evaluation_ok'])
    if d['evaluation_ok']:
        entry['cases']=d['cases']
    else:entry['error']=d.get('error')
    evaluations.append(entry)
assert len(evaluations)==24 and all(e['evaluation_ok'] for e in evaluations)
report=dict(run_dir=str(run),parent_dir=str(parent),completed_updates=500,
    step_mapping='Native runner TensorBoard iteration +1 = completed updates; Directions and eval already use completed updates.',
    windows=windows,evaluations=evaluations,
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    interpretation='Diminishing-return plateau in training objective, with residual improvement; not proof of full convergence. Late CPU 0.5m/s evaluations regress. Training and evaluation distributions differ; 3 paired starts per eval do not establish a population failure rate. Update-to-update SD is not multi-seed uncertainty.')
(out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'training_metrics.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['completed_update']+tags)
    for i in range(1,501):writer.writerow([i]+[series[t][i] for t in tags])
with (out/'evaluation_metrics.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['update','command','trials','failures','height_m','vx_m_s','yaw_rad_s'])
    for e in evaluations:
        for cmd,c in e['cases'].items():writer.writerow([e['update'],cmd,c['trials'],c['failures'],c.get('height_mean_m'),c.get('vx'),c.get('yaw')])

fig,axes=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
def curve(ax,tag,title):
    x=np.array(sorted(series[tag]));y=np.array([series[tag][i] for i in x])
    ax.plot(x,y,alpha=.25,lw=.8,color='#2278ad')
    ax.plot(x[19:],np.convolve(y,np.ones(20)/20,mode='valid'),color='#2278ad',label='20-update trailing mean')
    ax.set_title(title);ax.set_xlabel('Completed updates');ax.grid(alpha=.2)
curve(axes[0,0],'Train/mean_reward','Training episode reward');axes[0,0].set_ylim(20,29)
curve(axes[0,1],'Directions/forward/abs_velocity_error_0','Training forward |velocity error| (m/s)')
axes[0,1].set_ylim(0,.2)
for tag,label in [('Loss/value_function','Value loss'),('Loss/Estimation Loss','Estimator loss')]:
    x=sorted(series[tag]);y=[series[tag][i] for i in x]
    axes[1,0].semilogy(x,y,label=label,alpha=.8,lw=1)
axes[1,0].set_title('Training losses');axes[1,0].legend();axes[1,0].set_xlabel('Completed updates')
curve(axes[1,1],'Directions/forward/contact_slip_m_s_1','Training right-hind contact-slip diagnostic (m/s)')
colors={'stand':'#619c37','forward_035':'#277db5','forward_050':'#d17823'}
for cmd in colors:
    xs=[e['update'] for e in evaluations]
    ys=[e['cases'][cmd].get('height_mean_m',np.nan)*100 for e in evaluations]
    axes[2,0].plot(xs,ys,'o-',label=cmd,color=colors[cmd],markersize=3)
axes[2,0].axhline(30,ls='--',color='gray',lw=1)
axes[2,0].set_title('CPU evaluation body height (cm; pre-failure samples)');axes[2,0].legend()
for cmd in ['forward_035','forward_050']:
    axes[2,1].plot([e['update'] for e in evaluations],[e['cases'][cmd]['failures'] for e in evaluations],
        'o-',label=cmd,color=colors[cmd])
axes[2,1].set_title('CPU evaluation failures / 3 paired starts');axes[2,1].set_yticks([0,1,2,3]);axes[2,1].legend()
for ax in axes.flat:
    ax.axvline(400,color='gray',ls=':',lw=.9);ax.set_xlabel('Completed updates');ax.grid(alpha=.15)
fig.suptitle('New V2 posture training: reward plateau does not imply gait quality converged')
fig.savefig(out/'convergence.png',dpi=160);fig.savefig(out/'convergence.pdf');plt.close(fig)

def w(lo,hi):return next(r['metrics'] for r in windows if r['start']==lo and r['end']==hi)
before,after=w(301,400),w(401,500)
gain=(after['Train/mean_reward']['mean']/before['Train/mean_reward']['mean']-1)*100
lines=['# 新V2姿态修正500轮：收敛检查','',
'结论：当前奖励下已进入收益平台期，仍有小幅优化，不能认定所有指标完全收敛；后期0.5m/s评估出现退化。继续按原奖励加轮数，没有证据表明能解决右后擦地/步态不对称。', '',
'|指标|301–400轮均值|401–500轮均值|','|---|---:|---:|']
for label,tag in [('回合奖励','Train/mean_reward'),('直行速度绝对误差(m/s)','Directions/forward/abs_velocity_error_0'),
                  ('回合长度(步)','Train/mean_episode_length'),('价值损失','Loss/value_function'),('状态估计损失','Loss/Estimation Loss')]:
    lines.append('|%s|%.6f|%.6f|'%(label,before[tag]['mean'],after[tag]['mean']))
lines += ['', '两个100轮窗口的奖励均值仅提高%.3f%%。最后两个50轮窗口奖励从%.4f到%.4f，但直行速度误差从%.5f到%.5fm/s，未持续改善。'%(gain,w(401,450)['Train/mean_reward']['mean'],w(451,500)['Train/mean_reward']['mean'],w(401,450)['Directions/forward/abs_velocity_error_0']['mean'],w(451,500)['Directions/forward/abs_velocity_error_0']['mean']), '',
'## 独立姿态检查','', '|轮数|0.35m/s失败次数|0.5m/s失败次数|0.35m/s机身高度|','|---|---:|---:|---:|']
for e in evaluations:
    if e['update']>=300:
        c=e['cases'];lines.append('|%d|%d/3|%d/3|%.2fcm|'%(e['update'],c['forward_035']['failures'],c['forward_050']['failures'],c['forward_035']['height_mean_m']*100))
lines += ['', '0.5m/s失败序列420/440/460/480/500轮分别为2/1/1/3/1次（每次3个固定初始状态）。400轮这3次未失败，500轮并非所有评估指标最优。480轮3次都在统计窗口前失败，因此该点高度/速度缺失，不能插补为成功结果。', '',
'这些是相同CPU协议的固定3个起步状态，不是独立多种子训练，也不能将失败次数当作总体成功率。训练含探索和随机化，CPU检查为确定性策略、无随机化，不能将两者接触率直接混用。', '',
'建议：当前500轮保留为基线；不为解决步态问题直接延长原奖励训练。已获用户原则同意的摆腿离地和触地滑动奖励修改仍待实施。400轮可作为新的微调起点候选，先与500轮做同条件比较再选，不能仅凭总奖励或这3次评估认定最佳模型。']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(out=str(out),reward_window_gain_pct=gain),indent=2))
