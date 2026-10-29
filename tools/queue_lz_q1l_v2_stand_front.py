"""Detached single-job queue: preserve 4096 environments until enough VRAM is free."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from train_lz_q1l_v2_stand_front import ROOT, SOURCE, SOURCE_SHA256, preflight
from train_lz_q1l_v1_support_500 import write_json

SESSION = 'lz_q1l_v2_stand_front_300'
POINTER = ROOT / 'media/v2_stand_front_20261025/latest_training.json'
MIN_FREE_MIB = 12288  # Include headroom for PPO activations and shared-GPU fluctuations.


def retryable_startup_oom(run, code):
    status_path = run / 'status.json'
    if code == 0 or not status_path.exists():
        return False
    status = json.loads(status_path.read_text())
    return (status.get('state') == 'failed'
            and status.get('completed_updates') == 0
            and 'CUDA out of memory' in status.get('error', '')
            and not any(run.glob('model_*.pt')))


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    result = []
    for row in raw.strip().splitlines():
        index, uuid, free, util = [s.strip() for s in row.split(',')]
        result.append(dict(index=int(index), uuid=uuid, free_mib=int(free), utilization=int(util)))
    return result


def worker(meta_path):
    with (ROOT / '.cache/v2_stand_front_300.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        meta = json.loads(meta_path.read_text())
        run = Path(meta['run_dir'])
        original_run = run
        required_free = MIN_FREE_MIB
        attempts = []

        def persist(**changes):
            meta.update(changes, checked_at=datetime.datetime.now().isoformat(), worker_pid=os.getpid())
            write_json(meta_path, meta)
            write_json(POINTER, meta)

        try:
            preflight()
            while True:
                while True:
                    devices = gpu_snapshot()
                    selected = max(devices, key=lambda d: d['free_mib'])
                    persist(state='waiting_for_gpu', gpu_snapshot=devices, candidate_gpu=selected,
                            minimum_free_mib=required_free, selected_gpu=None, training_pid=None)
                    if selected['free_mib'] >= required_free:
                        break
                    print('WAITING_FOR_GPU', selected, 'required_mib', required_free, flush=True)
                    time.sleep(30)
                # Do not silently train changed code after waiting.
                for name, digest in meta['queued_code_sha256'].items():
                    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
                preflight()
                persist(state='starting', selected_gpu=selected)
                command = [sys.executable, '-u', str(ROOT / 'tools/train_lz_q1l_v2_stand_front.py'),
                           '--gpu', selected['uuid'], '--run-dir', str(run)]
                with Path(str(run) + '.log').open('x') as log:
                    process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
                    persist(state='training_process_started', training_pid=process.pid)
                    code = process.wait()
                Path(str(run) + '.exit_code').write_text(str(code) + '\n')
                if not retryable_startup_oom(run, code):
                    persist(state='completed' if code == 0 else 'failed', exit_code=code)
                    return code
                attempts.append(dict(run_dir=str(run), exit_code=code, gpu=selected,
                                     reason='CUDA OOM before first completed update'))
                required_free = min(16384, required_free + 1024)
                run = Path(str(original_run) + '_retry%d' % len(attempts))
                persist(state='waiting_for_gpu', run_dir=str(run), launch_log=str(run) + '.log',
                        failed_attempts=attempts, minimum_free_mib=required_free,
                        selected_gpu=None, training_pid=None)
                print('STARTUP_OOM_REQUEUED', attempts[-1], flush=True)
                time.sleep(30)
        except BaseException as exc:
            persist(state='queue_failed', error=repr(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    preflight()
    (ROOT / '.cache').mkdir(exist_ok=True)
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / '.cache/v2_stand_front_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (ROOT / '.cache/v2_stand_front_300.lock').open('a') as active:
            fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError('Gait session already exists')
        tag = 'stand_front_from_upright150_300_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid())
        run = ROOT / 'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_stand_front' / tag
        run.parent.mkdir(parents=True, exist_ok=True)
        parent_sources = json.loads((SOURCE.parent / 'run_metadata.json').read_text())['code_sha256']
        sources = list(parent_sources) + ['tools/train_lz_q1l_v2_stand_front.py',
            'tools/queue_lz_q1l_v2_stand_front.py', 'tools/start_lz_q1l_v2_stand_front_300.sh', 'tools/check_lz_q1l_stand_front.py']
        meta = dict(state='queued', run_dir=str(run), launch_log=str(run) + '.log',
            queue_status=str(run) + '.launch.json', session=SESSION, source_checkpoint=str(SOURCE),
            source_sha256=SOURCE_SHA256, source_iteration=150, additional_iterations=300,
            save_interval=50, checkpoint_iterations=[50, 100, 150, 200, 250, 300], auto_stop=False,
            minimum_free_mib=MIN_FREE_MIB, poll_interval_s=30, retry_startup_oom=True,
            queued_code_sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources})
        meta_path = Path(meta['queue_status'])
        write_json(meta_path, meta)
        if POINTER.exists():
            Path(str(run) + '.previous_pointer.json').write_text(POINTER.read_text())
        write_json(POINTER, meta)
        command = ['bash', str(ROOT / 'tools/start_lz_q1l_v2_stand_front_300.sh'), '--worker', str(meta_path)]
        try:
            subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, shlex.join(command)], check=True)
        except Exception as exc:
            meta.update(state='launch_failed', error=repr(exc))
            write_json(meta_path, meta)
            write_json(POINTER, meta)
            raise
        print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
