"""Render the completed standing diagnostic, without rerunning simulation."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

baseline, refined = map(Path, sys.argv[1:3])
records = []
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
colors = ['#218559', '#c63d3d', '#286ab6']
labels = ['Default PD, 5 ms', 'V8, 5 ms', 'V8, 2.5 ms']
for k, (directory, mode, index) in enumerate([
        (baseline, 'zero_action_pd', 0), (baseline, 'v8_zero_command', 33),
        (refined, 'v8_zero_command', 33)]):
    summary = json.loads((directory / 'summary.json').read_text())
    row = next(r for r in summary['cases'] if r['mode'] == mode and r['group'] == 'nominal')
    probe_summary = json.loads((directory / 'substep_summary.json').read_text())
    probe_row = next(r for r in probe_summary['nominal_cases'] if r['mode'] == mode)
    records.append(dict(label=labels[k], source=str(directory.resolve()),
        nominal=row, physics_substeps=probe_row))
    axes[0, 0].bar(np.arange(4)+(k-1)*.25, row['mean_fz_N'], .25,
                   color=colors[k], label=labels[k])
    trace = np.load(directory / 'traces.npz')
    t = (np.arange(len(trace['clearance']))+1)*.02
    axes[0, 1].plot(t, trace['clearance'][:, index, 1]*1000,
                   color=colors[k], label=labels[k], linewidth=1.5)
    probe = np.load(directory / 'physics_substeps.npz')
    dt = float(probe['dt'])
    time = np.arange(len(probe['data']))*dt
    window = (time >= 10) & (time <= 10.1)
    data = probe['data'][window, int(index > 0)]
    axes[1, 0].plot(time[window]-10, data[:, 1, 5], '.-', color=colors[k], label=labels[k])
    axes[1, 1].plot(time[window]-10, data[:, 3, 5], '.-', color=colors[k], label=labels[k])

axes[0, 0].set(xticks=np.arange(4), xticklabels=['Rear left', 'Rear right', 'Front left', 'Front right'],
    ylabel='Mean vertical contact force (N)', title='Nominal start: support force, 2-12 s')
axes[0, 0].legend(fontsize=9)
axes[0, 1].set(xlim=(2, 12), ylim=(-2, 55), xlabel='Time (s)',
    ylabel='Collision sphere bottom above floor (mm)', title='Rear-right foot: actual clearance')
axes[0, 1].axhline(0, color='gray', linewidth=.7)
axes[1, 0].set(xlabel='Time since 10 s (s)', ylabel='Angular velocity (rad/s)',
    title='Rear-right knee: every physics step')
axes[1, 1].set(xlabel='Time since 10 s (s)', ylabel='Commanded torque (N m)',
    title='Rear-right knee: torque alternation')
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.suptitle('V8 standing diagnostic | same checkpoint and gains | policy period = 20 ms', fontsize=14)
fig.savefig(refined / 'standing_comparison.png', dpi=170)
fig.savefig(refined / 'standing_comparison.pdf')
report = dict(records=records,
    findings=[
        'Default zero-action PD supports all four feet in the tested nominal and 32 perturbed starts.',
        'V8 at 5 ms physics step lifts the rear-right foot and has fast alternating rear-right joint velocity/torque.',
        'At 2.5 ms physics step, keeping the policy period 20 ms, nominal rear-right contact returns and rapid knee oscillation disappears.',
        'Support remains asymmetric; this standing test does not establish the cause of lateral/yaw training failure.'
    ], limitations=['Nominal flat-ground simulation; not hardware validation.',
        'Timestep changes the closed-loop trajectory; no proof that oscillation precedes foot lifting.',
        '32 perturbed starts use small initial velocities within +/-0.05, not a broad robustness test.'])
(refined / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
print(refined / 'standing_comparison.png')
