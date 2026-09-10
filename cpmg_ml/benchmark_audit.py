"""Read-only audit of existing shards; outputs a separate JSON report."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def audit(root):
    seen = {}
    counts = {}
    overlaps = {}
    invalid = {}
    excluded = {}
    manifests = []
    for split in ('train', 'val', 'test'):
        counts[split] = 0
        invalid[split] = 0
        excluded[split] = []
        for path in sorted(root.glob(f'{split}_*.h5')):
            with h5py.File(path) as h:
                arrays = [h[k][:] for k in ('metadata', 'nu_cp', 'r2_with_j', 'r2_no_j')]
                manifests.append({'file': str(path), 'rows': len(arrays[0]),
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'sequence': str(h.attrs.get('sequence', 'unknown')),
                                  't_relax': float(h.attrs['t_relax'])})
                for rows in zip(*arrays):
                    key = hashlib.sha256(b''.join(np.asarray(r, dtype='<f4').tobytes() for r in rows)).digest()
                    invalid[split] += int(not all(np.isfinite(r).all() for r in rows))
                    previous = seen.setdefault(key, set())
                    if previous:
                        excluded[split].append(counts[split])
                    for other in previous:
                        label = f'{other}->{split}'
                        overlaps[label] = overlaps.get(label, 0) + 1
                    previous.add(split)
                    counts[split] += 1
    return {'root': str(root), 'counts': counts, 'duplicate_occurrences_by_split': overlaps,
            'excluded_duplicate_row_indices': excluded,
            'nonfinite_rows': invalid, 'shards': manifests,
            'definition': 'Exact float32 metadata, axes and both curves; each duplicate row counted against each previously seen split. No files modified.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.data_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'shards'}, indent=2))
