from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from writepolicybench.episode_io import write_episodes_jsonl
from writepolicybench.synthetic import DriftConfig, generate_synthetic_drift_episodes


def main() -> int:
    out_dir = Path('data/episodes')
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 0
    steps = 200
    episodes_n = 10
    modes = ['default', 'burst_drift', 'redundancy', 'burst_redundancy']

    manifest = {
        'seed': seed,
        'steps': steps,
        'episodes': episodes_n,
        'modes': modes,
        'files': {},
    }

    for mode in modes:
        cfg = DriftConfig(steps=steps, seed=seed, mode=mode)
        episodes = generate_synthetic_drift_episodes(episodes_n, cfg)
        path = out_dir / f'episodes__schema=priority_v1__mode={mode}__seed={seed}__steps={steps}__n={episodes_n}.jsonl'
        write_episodes_jsonl(path, episodes)
        manifest['files'][mode] = str(path)

    (out_dir / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    print('Wrote', out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
