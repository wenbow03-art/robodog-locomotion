"""Aggregate the noise pilot's deterministic evaluations and export plots/CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trial_dir', type=Path)
    opts = parser.parse_args()
    root = opts.trial_dir
    rows, outcomes, metrics, manifests = [], {}, {}, {}
    for arm in ['fixed', 'annealed']:
        folder = root / arm
        outcomes[arm] = json.loads((folder / 'result.json').read_text())
        metrics[arm] = json.loads((folder / 'training_metrics.json').read_text())
        manifests[arm] = json.loads((folder / 'manifest.json').read_text())
        assert outcomes[arm]['updates'] == 50 and outcomes[arm]['source_unchanged']
        assert len(metrics[arm]) == 50
        for update in [10, 20, 30, 40, 50]:
            results = [json.loads((folder / ('eval_%03d' % update) / ('eval_checkpoint_seed%d.json' % seed)).read_text()) for seed in [1, 2]]
            for name in [c['case'] for c in results[0]['cases'] if c['mode'] == 'baseline']:
                row = dict(arm=arm, update=update, case=name)
                for mode in ['baseline', 'calibrated']:
                    cases = [next(c for c in result['cases'] if c['case']==name and c['mode']==mode) for result in results]
                    row[mode] = dict(command=cases[0]['command'],
                        mean_velocity=np.mean([c['mean'][:3] for c in cases], 0).tolist() if all('mean' in c for c in cases) else None,
                        velocity_sd_across_two_seed_means=np.std([c['mean'][:3] for c in cases], 0, ddof=1).tolist() if all('mean' in c for c in cases) else None,
                        mean_mae=np.mean([c['mae'] for c in cases], 0).tolist() if all('mae' in c for c in cases) else None,
                        failures=sum(c['failures'] for c in cases), trials=32,
                        valid_fraction=float(np.mean([c['valid_fraction'] for c in cases])))
                if row['baseline']['mean_mae'] is not None and row['calibrated']['mean_mae'] is not None:
                    row['mae_percent_change'] = [100*(b/a-1) if a else None for a,b in zip(row['baseline']['mean_mae'], row['calibrated']['mean_mae'])]
                rows.append(row)
    # Verify the intended experimental difference, not just the human labels.
    a, b = manifests['fixed'], manifests['annealed']
    for key in ['source_sha256', 'training_seed', 'num_envs', 'steps_per_update', 'total_updates',
                'learning_rate', 'learning_rate_schedule', 'load_optimizer', 'std_parameter_trainable', 'entropy_coef', 'code_sha256s']:
        assert a[key] == b[key], key
    for filename in ['env_config.json', 'train_config.json']:
        assert json.loads((root/'fixed'/filename).read_text()) == json.loads((root/'annealed'/filename).read_text()), filename
    assert all(m['factor']==1. for m in metrics['fixed'])
    assert metrics['annealed'][0]['factor']==1. and metrics['annealed'][-1]['factor']==.5
    assert max(m['behavior_log_prob_max_error'] for arm in metrics.values() for m in arm) <= 1e-3
    first_update_keys = ['value_loss','surrogate_loss','estimator_last_minibatch_loss',
                         'swap_last_minibatch_loss','mean_reward_per_step','failures','timeouts']
    first_update_matches = all(metrics['fixed'][0][k] == metrics['annealed'][0][k] for k in first_update_keys)
    assert first_update_matches
    unchanged = hashlib.sha256(Path(a['source']).read_bytes()).hexdigest()==a['source_sha256']
    assert unchanged
    summary = dict(
        outcomes=outcomes, comparisons=rows, source_unchanged=True,
        controlled_settings_verified=True, first_update_matches_exactly=first_update_matches,
        total_training_steps=10240000,
        training_seeds=[101], evaluation_seeds=[1,2], trials_per_case=32,
        protocol='12-second nominal flat-ground deterministic trials; 2-second warmup excluded from tracking; first failures counted over all 12 seconds; post-failure samples excluded.',
        limitations=['One training seed; two evaluation seeds are not independent training replications.',
                     'Only 50 updates per arm at 1024 environments; this is a pilot.',
                     'Training keeps V8 randomization; evaluation is nominal only.',
                     'The 0.2 m/s pure-lateral test is below the V8 dedicated command range [0.25,0.45] m/s; report it separately from in-range 0.4 m/s.',
                     'Same root reset velocities are paired across baseline/candidate; PhysX may introduce small differences.'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    with (root/'comparison.csv').open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['arm','update','case','mode','command_vx','command_vy','command_wz','mean_vx','mean_vy','mean_wz','mae_vx','mae_vy','mae_wz','failures','trials','valid_fraction'])
        for row in rows:
            for mode in ['baseline','calibrated']:
                c=row[mode]
                writer.writerow([row['arm'],row['update'],row['case'],mode,*c['command'],
                                 *(c['mean_velocity'] or [None]*3),*(c['mean_mae'] or [None]*3),c['failures'],c['trials'],c['valid_fraction']])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes=plt.subplots(2,3,figsize=(14,8))
        for arm,color in [('fixed','#2b6cb0'),('annealed','#c05621')]:
            selected=[r for r in rows if r['arm']==arm]
            for ax,case in zip(axes[0],['left_02','left_04','right_04']):
                series=[r for r in selected if r['case']==case]
                ax.plot([r['update'] for r in series],[r['calibrated']['mean_mae'][1] if r['calibrated']['mean_mae'] else np.nan for r in series],'-o',label=arm,color=color)
                ax.set_title(case+' lateral MAE (m/s)')
            series=[r for r in selected if r['case']=='forward_05']
            axes[1,0].plot([r['update'] for r in series],[r['calibrated']['mean_mae'][0] if r['calibrated']['mean_mae'] else np.nan for r in series],'-o',label=arm,color=color)
            axes[1,0].set_title('Forward 0.5: vx MAE (m/s)')
            axes[1,1].plot([10,20,30,40,50],[sum(r['calibrated']['failures'] for r in selected if r['update']==u and r['case'] in ['left_02','right_02','left_04','right_04']) for u in [10,20,30,40,50]],'-o',label=arm,color=color)
            axes[1,1].set_title('Lateral failures / 128 trials')
            axes[1,2].plot([m['update'] for m in metrics[arm]],[m['factor'] for m in metrics[arm]],label=arm,color=color)
            axes[1,2].set_title('Action std / original V8 std')
        for ax in axes.flat:
            ax.set_xlabel('Completed updates')
            ax.grid(alpha=.2)
            ax.legend()
        for ax in list(axes[0]) + [axes[1,0], axes[1,1]]:
            ax.set_ylim(bottom=0)
        for ax, case, axis in [(axes[0,0],'left_02',1), (axes[0,1],'left_04',1),
                               (axes[0,2],'right_04',1), (axes[1,0],'forward_05',0)]:
            baseline = next(r['baseline']['mean_mae'][axis] for r in rows if r['case']==case)
            ax.axhline(baseline, color='#555555', linestyle='--', linewidth=1, label='original V8')
            ax.legend()
        fig.suptitle('V8 noise pilot: deterministic evaluation, 2 evaluation seeds, 1 training seed')
        fig.tight_layout()
        fig.savefig(root/'comparison.png',dpi=160)
        plt.close(fig)
    except ImportError as error:
        print('Plot unavailable:',repr(error))
    for row in rows:
        if row['update']==50 and row['case'] in ['left_02','left_04','right_04','forward_05','stand']:
            print(json.dumps(row))
    print('SUMMARY_SAVED',root/'summary.json')


if __name__=='__main__':
    main()
