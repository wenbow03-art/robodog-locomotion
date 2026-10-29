"""Read-only windowed V4 convergence analysis; separate reward plateau from posture."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[1]
run=Path(json.loads((ROOT/'media/v4_lateral_20261027/latest_training.json').read_text())['run_dir'])
out=ROOT/'media/v4_convergence_20261027';out.mkdir(exist_ok=True)
events=EventAccumulator(str(run),size_guidance={'scalars':0});events.Reload()
series={}
for tag in events.Tags()['scalars']:
    if tag.endswith('/time') or tag.startswith('Perf/'):continue
    values={}
    for item in events.Scalars(tag):
        step=item.step if tag.startswith(('Directions/','V4/')) else item.step+1
        assert step not in values and np.isfinite(item.value),(tag,step)
        values[step]=item.value
    assert set(values)==set(range(1,501)),tag
    series[tag]=np.array([values[i] for i in range(1,501)])
rows=[json.loads(line) for line in (run/'direction_metrics.jsonl').read_text().splitlines()]
assert [r['update'] for r in rows]==list(range(1,501))
windows=[]
for lo in range(1,501,50):
    hi=lo+49
    scalars={key:dict(mean=float(values[lo-1:hi].mean()),std_across_updates=float(values[lo-1:hi].std()),slope_per50=float(np.polyfit(np.arange(50),values[lo-1:hi],1)[0]*50)) for key,values in series.items()}
    directions={}
    for direction in ['left','right']:
        ds=[r['directions'][direction] for r in rows[lo-1:hi]];weights=np.array([r['samples'] for r in ds])
        metrics={k:np.average([r[k] for r in ds],axis=0,weights=weights).tolist() for k in ['velocity','abs_velocity_error','force_z_N','contact']}
        contact_weights=np.asarray([r['contact'] for r in ds])*weights[:,None]
        metrics['contact_slip_m_s']=(np.sum(np.asarray([r['contact_slip_m_s'] for r in ds])*contact_weights,axis=0)/np.maximum(contact_weights.sum(0),1)).tolist()
        metrics['falls_per_1000_env_seconds']=1000*sum(r['fall_events'] for r in ds)/sum(r['exposure_env_seconds'] for r in ds)
        directions[direction]=metrics
    windows.append(dict(first=lo,last=hi,scalars=scalars,directions=directions))

a,b=windows[6],windows[9]
comparisons=[]
for label,key in [('总回合奖励','Train/mean_reward'),('价值损失','Loss/value_function'),('姿态惩罚绝对量','Episode/rew_orientation'),('侧移滑动惩罚绝对量','Episode/rew_lateral_feet_slide'),('侧移探索标准差','V4/lateral_std'),('PPO KL','V4/kl')]:
    x,y=a['scalars'][key]['mean'],b['scalars'][key]['mean']
    if '惩罚绝对量' in label:x,y=abs(x),abs(y)
    comparisons.append(dict(label=label,before=x,after=y,change_pct=100*(y/x-1)))
for direction,label in [('left','左移'),('right','右移')]:
    for key,idx,suffix in [('abs_velocity_error',1,'侧向速度MAE(m/s)'),('abs_velocity_error',0,'前后漂移MAE(m/s)'),('abs_velocity_error',2,'偏航角速度MAE(rad/s)'),('force_z_N',1 if direction=='left' else 2,'低承重脚平均力(N)'),('contact',1 if direction=='left' else 2,'低承重脚触地率')]:
        x,y=a['directions'][direction][key][idx],b['directions'][direction][key][idx]
        comparisons.append(dict(label=label+suffix,before=x,after=y,change_pct=100*(y/x-1)))

fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
x=np.arange(1,501)
panels=[(['Train/mean_reward'],['reward'],'Total episode reward'),
    (['Directions/left/abs_velocity_error_1','Directions/right/abs_velocity_error_1'],['left','right'],'Lateral velocity MAE (m/s)'),
    (['Directions/left/force_z_N_1','Directions/right/force_z_N_2'],['left: rear-right','right: front-left'],'Low-load foot mean force (N)'),
    (['Directions/left/contact_1','Directions/right/contact_2'],['left: rear-right','right: front-left'],'Low-load foot contact fraction'),
    (['Episode/rew_orientation'],['penalty magnitude'],'Whole-body orientation penalty magnitude'),
    (['V4/lateral_std'],['lateral action noise'],'Lateral exploration standard deviation')]
for ax,(tags,labels,title) in zip(axes.flat,panels):
    for tag,label in zip(tags,labels):
        y=series[tag];y=abs(y) if 'rew_orientation' in tag else y
        line=ax.plot(x,y,alpha=.18,lw=.7)[0]
        ax.plot(x[19:],np.convolve(y,np.ones(20)/20,mode='valid'),color=line.get_color(),label=label)
    ax.axvline(300,color='gray',ls=':',lw=1);ax.set_title(title);ax.set_xlabel('Completed updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('V4: reward plateaus after ~300 updates while support imbalance increases\nRaw values + trailing 20-update means; exploratory training, not deterministic evaluation')
fig.savefig(out/'convergence.png',dpi=160);fig.savefig(out/'convergence.pdf');plt.close(fig)

sources=list(run.glob('events.out.tfevents.*'))+[run/'direction_metrics.jsonl',run/'run_metadata.json',run/'status.json',Path(__file__)]
report=dict(run_dir=str(run),windows=windows,comparisons_301_350_to_451_500=comparisons,all_scalars_finite=True,
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    conclusion='Approximate reward/tracking plateau around update300; not complete convergence. Support imbalance continues worsening, with no sustained late reward gain. No evidence that unchanged continuation will fix posture.',
    limitations='Single training seed. Across-update SD is not independent uncertainty. Training contains exploration/domain randomization/transitions; cannot equate its forces/contact rates/speeds to deterministic replay. No standardized intermediate-checkpoint evaluation; cannot choose a best checkpoint or prove a local optimum from these curves. PPO surrogate/critic loss and KL are not proofs of task success.')
(out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'training_metrics.csv').open('w') as stream:
    writer=csv.writer(stream);writer.writerow(['update']+list(series))
    for i in range(500):writer.writerow([i+1]+[float(v[i]) for v in series.values()])
lines=['# 新v4 500轮收敛检查','',
    '**总奖励和速度跟踪在约300轮后基本进入平台；不能说所有指标完全收敛，姿态与支撑质量仍在变差。**','',
    '|指标|301～350轮均值|451～500轮均值|相对变化|','|---|---:|---:|---:|']
for r in comparisons:lines.append('|%s|%.6f|%.6f|%+.2f%%|'%(r['label'],r['before'],r['after'],r['change_pct']))
lines+=['','## 判断','',
    '1. 总奖励301～350轮38.0547，351～400轮38.0230，401～450轮37.7655，451～500轮37.8473；最后200轮没有持续上升。最后两个50轮窗口仅增加约0.22%，小于前期提升幅度。',
    '2. 侧移速度误差在最后阶段基本不再降低。左移低承重的右后脚平均力22.79→19.90N，触地29.83%→25.03%；右移低承重的左前脚10.58→6.11N，触地25.70%→13.45%。支撑不均继续加重。',
    '3. 全身姿态惩罚绝对量约增加42%；价值损失没有数值发散。不能把小的PPO代理损失或较长回合当作步态合格。',
    '4. 侧移探索标准差后期仍变化。日志通用的Policy/mean_noise_std约0.11是被冻结的原std，判断侧移探索必须使用V4/lateral_std（最后50轮约0.292）。',
    '5. 训练方向日志的velocity第三项为偏航角速度，不是竖直线速度。contact_slip_m_s是按接触时间归一化的滑速；图表与统计保持该口径。',
    '6. 训练有随机化、探索和指令切换，这里的触地率/力/速度不能与无噪声固定指令视频直接混比。单训练种子的曲线平台不证明严格收敛、最优或将来永远无改进。',
    '', '## 建议','',
    '暂不按原配置盲目续训。先在相同无镜像固定指令下比较model300、400、500，确认后期偏载是否随检查点加重，再决定侧移专用奖惩如何调整。不能凭奖励最高就认定model300最好。本次仅分析日志，未启动训练或修改参数。']
(out/'findings_zh.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(comparisons=comparisons,plot=str(out/'convergence.png'),all_finite=True),indent=2))
