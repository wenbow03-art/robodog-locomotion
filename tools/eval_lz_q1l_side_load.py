"""Matched frozen-policy comparison after the one-sided load experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from train_lz_q1l_v2_side_load import ROOT, SOURCE, SOURCE_SHA256


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    run = Path(sys.argv[1]).resolve()
    out = run / 'evaluation'
    out.mkdir(exist_ok=False)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                  '--format=csv,noheader,nounits'], text=True)
    devices = [(r.split(',')[0].strip(), int(r.split(',')[1])) for r in raw.strip().splitlines()]
    gpu = max(devices, key=lambda v: v[1])[0]
    process_env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    models = [('baseline150', SOURCE, 150), ('leftload50', run/'model_50.pt', 50),
              ('leftload100', run/'model_100.pt', 100)]
    save(out/'status.json', dict(state='running', gpu=gpu))
    try:
        data = {}
        for label, path, iteration in models:
            target = out/label
            target.mkdir()
            save(target/'models.json', [dict(label=label, path=str(path), iteration=iteration)])
            command = [sys.executable, '-u', str(ROOT/'tools/audit_lz_q1l_new_lineage.py'),
                       '--models', str(target/'models.json'), '--out', str(target/'eval'),
                       '--command-seconds', '20', '--task', 'lz_q1l_flat_v2_balance_roll',
                       '--headless', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--seed', '1']
            with (target/'eval.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=process_env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            with (target/'analysis.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT/'tools/summarize_lz_q1l_lineage_audit.py'),
                                str(target)], cwd=ROOT, env=process_env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            data[label] = json.loads((target/'analysis.json').read_text())
            data[label]['trials'] = json.loads((target/'eval/report.json').read_text())['trials']
            print('EVALUATED', label, flush=True)
        lookup = {(label, g['speed']): g for label, d in data.items() for g in d['groups']}
        rows = []
        lines = ['# 单边左侧过载惩罚：与原始 gait150 比较', '',
                 '各模型独立使用同样12环境布局、3个配对初速度起点、4种指令；预站2秒、指令20秒，统计指令第1–20秒。关闭噪声及域随机化。3个起点不是3个训练种子。', '',
                 '|模型|指令m/s|左侧承重%|距50%的偏差（百分点）|侧倾°|后腿力差N|前腿力差N|高度cm|速度MAE|失败数|',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        metrics = ['left_share_pct', 'roll_abs_mean_deg', 'rear_force_difference_n',
                   'front_force_difference_n', 'height_cm', 'vx_mae', 'all_four_contact_pct',
                   'force_n', 'contact_pct', 'gap_max_mm', 'loaded_slip_m_s']
        for speed in [0., .2, .35, .5]:
            for label, _, _ in models:
                g = lookup[label, speed]
                row = dict(model=label, command=speed, failures=g['failures'])
                for metric in metrics:
                    row[metric] = g[metric]
                # Also retain per-start errors: averaging opposite biases can hide imbalance.
                trials = [t for t in data[label]['trials'] if t.get('command') == speed]
                shares = [100.*(t['force_n'][0]+t['force_n'][2])/sum(t['force_n'])
                          for t in trials if 'force_n' in t and sum(t['force_n']) > 0]
                errors = [abs(s-50.) for s in shares]
                row['per_start_left_share_pct'] = shares
                row['mean_per_start_side_share_error_pp'] = sum(errors)/len(errors) if errors else None
                row['side_share_error_from_group_mean_pp'] = abs(g['left_share_pct']['mean'] - 50.)
                base = lookup['baseline150', speed]
                row['delta_vs_baseline'] = {m: g[m]['mean']-base[m]['mean'] for m in metrics[:7]}
                rows.append(row)
                lines.append('|%s|%.2f|%.2f|%.2f|%.2f|%.2f|%.2f|%.2f|%.4f|%d/3|' % (
                    label, speed, g['left_share_pct']['mean'], row['side_share_error_from_group_mean_pp'],
                    g['roll_abs_mean_deg']['mean'], g['rear_force_difference_n']['mean'],
                    g['front_force_difference_n']['mean'], g['height_cm']['mean'], g['vx_mae']['mean'], g['failures']))
        lines += ['', '新增项仅惩罚左侧过重，右侧过重为零，因此惩罚降低不能单独证明左右恢复平衡。需要同时检查左右占比、各起点结果、侧倾、前后腿配对力差和步态。',
                  '这是冻结策略对比；没有相同轮数的原配置续训对照，不能把全部变化单独归因于新增奖励。']
        save(out/'comparison.json', dict(rows=rows, models=data))
        (out/'report.md').write_text('\n'.join(lines)+'\n')
        save(out/'status.json', dict(state='completed', gpu=gpu, report=str(out/'report.md')))
        print('\n'.join(lines), flush=True)
    except BaseException as exc:
        save(out/'status.json', dict(state='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
