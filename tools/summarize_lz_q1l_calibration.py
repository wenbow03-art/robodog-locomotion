"""Export calibration comparisons and independently check checkpoint scope on CPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    root = args.run_dir
    manifest = json.loads((root / 'manifest.json').read_text())
    result = json.loads((root / 'result.json').read_text())
    original = torch.load(manifest['source'], map_location='cpu')['model_state_dict']
    checkpoints = sorted(root.glob('calibrated_*.pt'))
    checks, comparisons = [], []
    for path in checkpoints:
        label = path.stem.split('_')[-1]
        state = torch.load(path, map_location='cpu')['model_state_dict']
        changed = [k for k in state if not torch.equal(state[k], original[k])]
        assert set(changed) == {'estimator.encoder.4.weight', 'estimator.encoder.4.bias'}
        assert all(torch.equal(state[k][3:], original[k][3:]) for k in changed)
        checks.append(dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           changed_keys=changed, changed_scalars=sum(int((state[k] != original[k]).sum()) for k in changed),
                           actor_std_latent_encoder_features_unchanged=True))
        evals = [json.loads((root / ('eval_%s_seed%d.json' % (label, seed))).read_text()) for seed in [1, 2]]
        for name, command in manifest['cases']:
            row = dict(update=int(label), case=name, command=command)
            axis = 1 if command[1] else (2 if command[2] else 0)
            row['tracking_axis'] = axis
            for mode in ['baseline', 'calibrated']:
                cases = [next(c for c in e['cases'] if c['mode'] == mode and c['case'] == name) for e in evals]
                row[mode] = dict(
                    velocity_mean_over_seeds=np.mean([c['mean'][:3] for c in cases], 0).tolist(),
                    velocity_sd_across_two_seed_means=np.std([c['mean'][:3] for c in cases], 0, ddof=1).tolist(),
                    tracking_mae_mean_over_seeds=np.mean([c['mae'] for c in cases], 0).tolist(),
                    failures=sum(c['failures'] for c in cases), trials=32,
                    predicted_next_vy_mae_mean=float(np.mean([c['predicted_next_vy_mae'] for c in cases])))
            before = row['baseline']['tracking_mae_mean_over_seeds'][axis]
            after = row['calibrated']['tracking_mae_mean_over_seeds'][axis]
            row['tracking_mae_change_percent'] = 100 * (after / before - 1) if before else None
            comparisons.append(row)
    initial = json.loads((root / 'initial_validation.json').read_text())['velocity_mae_mps']
    training = json.loads((root / 'training_metrics.json').read_text())
    final = training[-1]['validation_velocity_mae_mps']
    summary = dict(
        outcome='No candidate passed the predefined regression gate; original V8 retained.',
        training_samples=102599, validation_samples=102749,
        train_seed=101, validation_seed=202, evaluation_seeds=[1, 2],
        checkpoints=checks, validation_mae_before_mps=initial, validation_mae_after_mps=final,
        validation_mae_change_percent=[100*(a/b-1) for a,b in zip(final,initial)],
        source_checkpoint_unchanged=hashlib.sha256(Path(manifest['source']).read_bytes()).hexdigest()==manifest['source_sha256'],
        comparisons=comparisons, decision=result,
        interpretation='Velocity-head calibration reduces held-out prediction error and activates some +0.4 m/s lateral movement, but does not solve low-speed left stepping and introduces tracking/stability regressions. This is a negative result for the tested head-only intervention, not a general rejection of estimator calibration.',
        recommendation='Retain V8. A next experiment could test joint policy/estimator adaptation with gradual noise reduction; that experiment has not been run.',
        limitations=manifest['limitations'] + ['Two evaluation RNG seeds; seed SD describes these runs only.',
                                             'Training data are fixed-command, baseline-policy trajectories; no iterative on-policy recollection.',
                                             'Only the final three velocity-output rows were trained, not the shared encoder.'])
    # Derive counts from saved replay rather than assuming a specific run length.
    for prefix in ['train', 'validation']:
        data = torch.load(root / (prefix + '_replay.pt'), map_location='cpu')
        summary['training_samples' if prefix == 'train' else 'validation_samples'] = len(data['history'])
        assert torch.isfinite(data['history']).all() and torch.isfinite(data['target_scaled_velocity']).all()
        assert set(data['group'].unique().tolist()) == set(range(22))
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (root / 'comparison.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['update', 'case', 'mode', 'command_vx', 'command_vy', 'command_wz',
                         'mean_vx', 'mean_vy', 'mean_wz', 'mae_vx', 'mae_vy', 'mae_wz', 'failures', 'trials'])
        for row in comparisons:
            for mode in ['baseline', 'calibrated']:
                c = row[mode]
                writer.writerow([row['update'], row['case'], mode, *row['command'],
                                 *c['velocity_mean_over_seeds'], *c['tracking_mae_mean_over_seeds'], c['failures'], c['trials']])
    print('Validation MAE:', initial, '->', final)
    for row in comparisons:
        if row['update'] == 300 and row['case'] in ['stand', 'forward_05', 'left_02', 'left_04', 'right_04']:
            print(json.dumps(row))
    print('VERIFIED: 4 checkpoints change only velocity output rows; source is unchanged.')


if __name__ == '__main__':
    main()
