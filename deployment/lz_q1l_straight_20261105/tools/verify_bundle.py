#!/usr/bin/env python3
"""Verify package integrity; optionally run CPU TorchScript synthetic inputs."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_integrity():
    entries = (ROOT / 'SHA256SUMS').read_text().splitlines()
    if not entries:
        raise ValueError('Empty checksum manifest')
    for entry in entries:
        expected, relative = entry.split('  ', 1)
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or digest(path) != expected:
            raise ValueError('Checksum mismatch or invalid path: ' + relative)
    return len(entries)


def verify_model():
    import torch
    torch.set_num_threads(1)
    c = json.loads((ROOT / 'config/policy_contract.json').read_text())
    if digest(ROOT / c['model']) != c['model_sha256']:
        raise ValueError('Policy digest differs from the deployment contract')
    policy = torch.jit.load(str(ROOT / c['model']), map_location='cpu').eval()
    scale = torch.tensor(c['observation_layout'][0]['scale'])
    if not torch.equal(policy.scale, scale):
        raise ValueError('Command scales differ from the exported policy')
    # Synthetic standing sensor values. This checks the interface, not dynamics.
    sequence = [(0., 50), (.5, 30), (.75, 30), (1., 30), (1.25, 30),
                (1.5, 30), (2., 30), (.4, 30), (0., 30)]

    def run():
        history = torch.zeros(1, 270)
        ids = torch.ones(1, 14, dtype=torch.long)
        alphas = torch.zeros(1, 2)
        previous = torch.zeros(1, 12)
        traces = []
        with torch.no_grad():
            for vx, steps in sequence:
                for _ in range(steps):
                    frame = torch.zeros(1, 45)
                    frame[:, 0] = vx * scale[0]
                    frame[:, 8] = -1.
                    frame[:, 33:45] = previous
                    history = torch.cat((frame, history[:, :225]), dim=1).clamp(-100., 100.)
                    errors = torch.tensor([[.02, -.01, .03]]) if vx > 0 else torch.zeros(1, 3)
                    action, ids, alphas = policy(history, errors, ids, alphas)
                    if action.shape != (1, 12) or ids.shape != (1, 14) or alphas.shape != (1, 2):
                        raise ValueError('Wrong output shapes')
                    if action.dtype != torch.float32 or alphas.dtype != torch.float32 or ids.dtype != torch.int64:
                        raise ValueError('Wrong output dtypes')
                    if not torch.isfinite(action).all() or not torch.isfinite(alphas).all():
                        raise ValueError('Nonfinite output')
                    if not ((ids >= 0) & (ids <= 3)).all() or not ((alphas >= 0) & (alphas <= 1)).all():
                        raise ValueError('Invalid routing/blend state')
                    previous = action.clamp(-100., 100.)
                    traces.append(action.clone())
        if not torch.equal(ids, torch.ones_like(ids)) or not torch.equal(alphas, torch.zeros_like(alphas)):
            raise ValueError('Standing did not reset routing/blend state after settling')
        return torch.stack(traces)

    first, second = run(), run()
    if not torch.equal(first, second):
        raise ValueError('Reset/replay was not deterministic')
    return {'torch_version': torch.__version__, 'device': 'cpu',
            'steps_per_run': len(first), 'runs': 2,
            'reset_replay_exact': True, 'all_outputs_finite': True,
            'forward_schema': str(policy.forward.schema),
            'scope': 'synthetic interface check; no simulated or physical robot'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='store_true', help='Also require PyTorch and run the CPU check')
    args = parser.parse_args()
    result = {'files_verified': verify_integrity(), 'python': platform.python_version(),
              'platform': platform.machine(), 'hardware_ready': False}
    if args.model:
        result['model_check'] = verify_model()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
