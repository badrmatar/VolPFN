"""Make a separate pickle-free archive, preserving the original bootstrap file.

Only labels are exported again from the verified common-mask CSV. No draws,
gains, counts, statistics or scientific results are regenerated or modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(folder):
    folder = Path(folder)
    source = folder / 'bootstrap_archive.npz'
    destination = folder / 'bootstrap_archive_no_pickle.npz'
    receipt = folder / 'pickle_free_export_receipt.json'
    if destination.exists() or receipt.exists():
        raise FileExistsError('Refuse to overwrite either archive or receipt')
    before = sha(source)
    mask_path = folder / 'common_mask.csv'
    mask = pd.read_csv(mask_path, dtype={'market': str, 'date': str})
    # Never deserialize the object-typed original labels; use the mask instead.
    with np.load(source, allow_pickle=False) as old:
        arrays = {key: old[key] for key in old.files if key not in ['row_dates', 'row_markets']}
    arrays['row_dates'] = mask.date.to_numpy(dtype='U10')
    arrays['row_markets'] = mask.market.to_numpy(dtype=str)
    for key, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError('Object array remains: ' + key)
    with destination.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    with np.load(source, allow_pickle=False) as old, np.load(destination, allow_pickle=False) as new:
        if set(old.files) != set(new.files):
            raise ValueError('Archive fields changed')
        for key in old.files:
            if key not in ['row_dates', 'row_markets']:
                np.testing.assert_array_equal(old[key], new[key], err_msg='Scientific array changed: ' + key)
        np.testing.assert_array_equal(new['row_dates'], mask.date)
        np.testing.assert_array_equal(new['row_markets'], mask.market)
    if sha(source) != before:
        raise ValueError('Original archive changed')
    with receipt.open('x') as handle:
        json.dump({'status': 'passed', 'original_unchanged_sha256': before,
            'safe_archive_sha256': sha(destination), 'common_mask_sha256': sha(mask_path),
            'export_source_sha256': sha(__file__), 'numeric_arrays_bit_identical': True,
            'all_arrays_readable_without_pickle': True, 'labels_from_verified_common_mask': True}, handle, indent=2)
        handle.write('\n')
    print('Pickle-free archive verified; original preserved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    export(parser.parse_args().folder)
