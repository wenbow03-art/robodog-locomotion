"""Export a V8 run's scalar data and curves without requiring a GPU."""
import argparse
import csv
import json
from pathlib import Path


def export_curves(run_dir, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir, output_dir = Path(run_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = EventAccumulator(str(run_dir), size_guidance={'scalars': 0})
    events.Reload()
    tags = events.Tags()['scalars']
    metrics = [
        ('Train/mean_reward', 'Mean reward'),
        ('Train/mean_episode_length', 'Episode length (steps)'),
        ('Episode/rew_tracking_lateral', 'Lateral tracking reward'),
        ('Episode/rew_lateral_error', 'Lateral error penalty (weighted)'),
        ('Episode/rew_tracking_lin_vel', 'Planar velocity tracking reward'),
        ('Episode/rew_tracking_ang_vel', 'Yaw tracking reward'),
        ('Loss/value_function', 'Value loss'),
        ('Policy/mean_noise_std', 'Action noise std'),
    ]
    missing = [tag for tag, _ in metrics if tag not in tags]
    if missing:
        raise RuntimeError('Missing training metrics: ' + ', '.join(missing))
    prefix = output_dir / run_dir.name
    csv_path = Path(str(prefix) + '_scalars.csv')
    with csv_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['tag', 'tensorboard_step', 'completed_v8_updates', 'value', 'wall_time'])
        for tag in tags:
            if tag.endswith('/time'):
                continue
            for event in events.Scalars(tag):
                writer.writerow([tag, event.step, event.step + 1, event.value, event.wall_time])
    figure, axes = plt.subplots(4, 2, figsize=(13, 13))
    for axis, (tag, title) in zip(axes.flat, metrics):
        values = events.Scalars(tag)
        x, y = [v.step + 1 for v in values], [v.value for v in values]
        smoothed = []
        for value in y:
            smoothed.append(value if not smoothed else 0.9 * smoothed[-1] + 0.1 * value)
        axis.plot(x, y, color='#2474b5', alpha=0.25, label='Raw')
        axis.plot(x, smoothed, color='#2474b5', label='EMA 0.9')
        for step in (50, 100, 150, 200):
            axis.axvline(step, color='#d97706', linestyle='--', linewidth=0.7, alpha=0.5)
        axis.set_title(title)
        axis.set_xlabel('Completed V8 updates (initialized from V7 model_200)')
        axis.grid(alpha=0.15)
    axes[0, 0].legend()
    figure.suptitle(run_dir.name, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    png_path = Path(str(prefix) + '_training_curves.png')
    figure.savefig(str(png_path), dpi=160)
    plt.close(figure)
    result = {'run_dir': str(run_dir), 'png': str(png_path), 'csv': str(csv_path),
              'step_mapping': 'completed_v8_updates = tensorboard_step + 1'}
    Path(str(prefix) + '_curve_sources.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('/home/xiaohengli/Program/training-curves'))
    args = parser.parse_args()
    print(json.dumps(export_curves(args.run_dir, args.output_dir), indent=2))
