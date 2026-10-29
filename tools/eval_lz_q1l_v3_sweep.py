"""Matched final-checkpoint evaluation of five other-leg preservation weights."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from train_lz_q1l_v3_rear_teacher import ROOT, SOURCE, SOURCE_SHA256
from train_lz_q1l_v1_support_500 import write_json


def main():
    manifest=Path(sys.argv[1]).resolve();meta=json.loads(manifest.read_text())
    out=manifest.parent/'evaluation';out.mkdir(exist_ok=False)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256
    output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    devices=[(r.split(',')[0].strip(),int(r.split(',')[1])) for r in output.strip().splitlines()]
    gpu=max(devices,key=lambda x:x[1])[0];process_env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
    models=[dict(label='baseline150',path=str(SOURCE),iteration=150,coefficient=None,sha256=SOURCE_SHA256)]
    for job in meta['jobs']:
        if job['state']=='completed':
            path=Path(job['run_dir'])/('model_%d.pt'%meta['iterations'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==job['checkpoint_sha256']
            models.append(dict(label='v3_other%.2f'%job['other_legs_coef'],path=str(path),iteration=meta['iterations'],
                               coefficient=job['other_legs_coef'],sha256=job['checkpoint_sha256']))
    write_json(out/'status.json',dict(state='running',gpu=gpu,models=models))
    try:
        results={}
        for model in models:
            target=out/model['label'];target.mkdir()
            write_json(target/'models.json',[model])
            command=[sys.executable,'-u',str(ROOT/'tools/audit_lz_q1l_new_lineage.py'),
                     '--models',str(target/'models.json'),'--out',str(target/'eval'),
                     '--command-seconds','20','--task','lz_q1l_flat_v2_balance_roll',
                     '--headless','--sim_device','cuda:0','--rl_device','cuda:0','--seed','1']
            with (target/'eval.log').open('x') as log:
                subprocess.run(command,cwd=str(ROOT),env=process_env,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (target/'analysis.log').open('x') as log:
                subprocess.run([sys.executable,str(ROOT/'tools/summarize_lz_q1l_lineage_audit.py'),str(target)],
                               cwd=str(ROOT),env=process_env,stdout=log,stderr=subprocess.STDOUT,check=True)
            results[model['label']]=json.loads((target/'analysis.json').read_text())
            print('EVALUATED',model['label'],flush=True)
        rows=[]
        lines=['# 新 v3：其他三腿保持权重扫描','',
               '右后腿监督权重固定 0.10；其他三腿权重 0/0.01/0.02/0.03/0.04。各组从同一 gait150 训练 300 轮，种子相同；无自动质量终止。',
               '每个冻结模型独立回放同样 12 个环境、3 个初速度起点、4 种指令；预站 2 秒、指令 20 秒，统计指令第 1–20 秒；关闭噪声和域随机化。',
               '三个起点不是三个训练种子。baseline150 未续训；权重 0 组仍有右后腿监督，并非原 PPO 续训对照。','',
               '|模型|速度m/s|失败/3|左侧承重%|侧倾绝对值°|后腿力差N|前腿力差N|右后脚最大离地mm|右后脚触地%|速度误差m/s|',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for speed in (0.,.2,.35,.5):
            for model in models:
                group=next(g for g in results[model['label']]['groups'] if g['speed']==speed)
                rows.append(dict(label=model['label'],coefficient=model['coefficient'],**group))
                def scalar(key,index=None):
                    value=group.get(key,{}).get('mean')
                    if index is not None and value is not None:value=value[index]
                    return 'NA' if value is None else '%.3f'%value
                lines.append('|%s|%.2f|%d|%s|%s|%s|%s|%s|%s|%s|'%(model['label'],speed,group['failures'],
                             scalar('left_share_pct'),scalar('roll_abs_mean_deg'),scalar('rear_force_difference_n'),
                             scalar('front_force_difference_n'),scalar('gap_max_mm',1),scalar('contact_pct',1),scalar('vx_mae')))
        lines+=['','比较时同时看四脚承重、前后腿动作、侧倾、滑动和速度，不以监督误差或训练总奖励单独判断优劣。',
                '各组 analysis.json 包含实际关节角、目标关节角与跟踪误差；eval/traces.npz 保留完整时序，供进一步核对前腿是否退化。']
        write_json(out/'comparison.json',dict(models=models,rows=rows,results=results))
        (out/'report.md').write_text('\n'.join(lines)+'\n')
        write_json(out/'status.json',dict(state='completed',gpu=gpu,report=str(out/'report.md')))
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':main()
