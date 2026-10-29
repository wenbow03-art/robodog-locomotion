"""Read-only convergence analysis of completed gait200, not current height training."""
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


def main():
    run=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_gait/gait_from_posture500_200_20261025_113614_3235946'
    out=ROOT/'media/gait200_convergence_20261025';out.mkdir(exist_ok=False)
    status=json.loads((run/'status.json').read_text())
    assert status['state']=='completed' and status['completed_updates']==200
    ea=EventAccumulator(str(run),size_guidance={'scalars':0});ea.Reload()
    tags=[t for t in ea.Tags()['scalars'] if not t.endswith('/time') and not t.startswith('Perf/')]
    series={}
    for tag in tags:
        values={}
        for item in ea.Scalars(tag):
            step=item.step+(0 if tag.startswith('Directions/') else 1)
            assert step not in values and np.isfinite(item.value),(tag,step)
            values[step]=item.value
        assert set(values)==set(range(1,201)),tag
        series[tag]=values
    windows=[]
    for lo,hi in [(1,50),(51,100),(101,150),(151,200),(101,125),(126,150),(151,175),(176,200)]:
        metrics={}
        for tag,values in series.items():
            x=np.arange(lo,hi+1);y=np.array([values[i] for i in x])
            metrics[tag]=dict(mean=float(y.mean()),std_across_updates=float(y.std()),
                              slope_per50_updates=float(np.polyfit(x,y,1)[0]*50))
        windows.append(dict(start=lo,end=hi,metrics=metrics))
    def w(lo,hi):return next(v['metrics'] for v in windows if v['start']==lo and v['end']==hi)
    before,after=w(101,150),w(151,200)
    chosen=[('总回合奖励','Train/mean_reward'),('直行速度绝对误差','Directions/forward/abs_velocity_error_0'),
        ('摆腿离地惩罚','Episode/rew_swing_clearance'),('触地滑动惩罚','Episode/rew_feet_slide'),
        ('后腿力平衡惩罚','Episode/rew_rear_force_balance'),('身高偏差惩罚','Episode/rew_base_height'),
        ('站立离地惩罚','Episode/rew_stand_unsupported'),('价值损失','Loss/value_function'),
        ('状态估计损失','Loss/Estimation Loss'),('Swap损失','Loss/Swap Loss'),
        ('PPO代理损失','Loss/surrogate'),('回合长度','Train/mean_episode_length')]
    comparisons=[]
    for label,tag in chosen:
        a,b=before[tag]['mean'],after[tag]['mean']
        comparisons.append(dict(metric=label,tag=tag,before=a,after=b,
            absolute_magnitude_change_pct=float((abs(b)/abs(a)-1)*100) if a else None))
    sources=list(run.glob('events.out.tfevents.*'))+[run/'status.json',run/'run_metadata.json',run/'direction_metrics.jsonl']
    report=dict(run_dir=str(run),completed_updates=200,scalar_tags_checked=len(tags),all_finite=True,
        step_mapping='Native iteration +1 = completed update. Directions already use completed updates. Time-axis tags excluded.',
        windows=windows,comparisons=comparisons,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        interpretation='Total reward has diminishing gains; gait penalties continue improving. Height and standing-support penalties regress between the final 50-update windows. Not full convergence.',
        limitations='One training seed. Update-to-update SD is not independent uncertainty. Training includes exploration, randomized physics/starts and command transitions; not comparable directly to deterministic video contact rates. No intermediate-checkpoint standardized evaluation in this run. PPO surrogate need not monotonically decrease or approach zero.')
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'training_metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['completed_update']+tags)
        for i in range(1,201):writer.writerow([i]+[series[t][i] for t in tags])
    panels=[('Train/mean_reward','Episode reward',False),
        ('Directions/forward/abs_velocity_error_0','Forward absolute speed error (m/s)',False),
        ('Episode/rew_swing_clearance','Swing-clearance penalty magnitude',True),
        ('Episode/rew_feet_slide','Foot-slide penalty magnitude',True),
        ('Episode/rew_rear_force_balance','Rear-force balance penalty magnitude',True),
        ('Episode/rew_base_height','Height penalty magnitude',True),
        ('Episode/rew_stand_unsupported','Standing unsupported penalty magnitude',True),
        ('Loss/Estimation Loss','Estimator loss',False)]
    fig,axes=plt.subplots(4,2,figsize=(12,13),constrained_layout=True)
    for ax,(tag,title,absolute) in zip(axes.flat,panels):
        x=np.arange(1,201);y=np.array([series[tag][i] for i in x]);y=abs(y) if absolute else y
        ax.plot(x,y,alpha=.25,lw=.8,color='#247c8f')
        ax.plot(x[9:],np.convolve(y,np.ones(10)/10,mode='valid'),color='#247c8f',label='10-update trailing mean')
        ax.axvline(150,color='gray',ls=':',lw=1);ax.set_title(title);ax.set_xlabel('Completed gait updates');ax.grid(alpha=.2)
        if tag=='Train/mean_reward':ax.set_ylim(26.5,28.4)
        if tag=='Loss/Estimation Loss':ax.set_ylim(.0048,.007)
    axes[0,0].legend();fig.suptitle('Gait200: reward plateau, residual gait gains, standing/height trade-offs')
    fig.savefig(out/'convergence.png',dpi=150);fig.savefig(out/'convergence.pdf');plt.close(fig)
    lines=['# 步态修正200轮收敛检查','',
        '结论：总奖励接近平台，但不能说各项已收敛。后期离地、滑动、后腿受力仍有改善；身高与站立离地指标发生反复。','',
        '|指标|101–150轮均值|151–200轮均值|绝对量变化|','|---|---:|---:|---:|']
    for r in comparisons:
        lines.append('|%s|%.7f|%.7f|%+.2f%%|'%(r['metric'],r['before'],r['after'],r['absolute_magnitude_change_pct']))
    lines+=['','惩罚列保留日志负号；其绝对值下降代表代价减小。总奖励上升代表改善。',
        '最后两个25轮窗口（151–175→176–200）：总奖励27.9527→28.0416；摆腿惩罚绝对量下降约2.1%，滑动下降约5.0%，后腿平衡下降约13.0%，仍有优化趋势。',
        '价值损失最后两个50轮窗口近似不变；估计损失下降约2.5%，Swap损失下降约2.4%，均为有限值，无发散迹象。PPO代理损失波动不能按监督学习损失判断是否收敛。',
        '身高惩罚绝对量增加约19.2%；站立离地惩罚从0.000116增到0.000389，绝对数值小但增加约235%。',
        '训练中站立右后触地率均值67.36%→63.53%，后腿平均受力差约47.38→53.48 N。训练含探索、随机化、动作切换和起步；不能将此比例直接等同于视频固定零指令下的触地率。',
        '当前新增300轮使用了更强高度项和新的站立承重目标，因此是在改变优化目标，而非简单重复上一轮；本检查没有中止或改变当前训练。',
        '',report['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(out=str(out),comparisons=comparisons),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
