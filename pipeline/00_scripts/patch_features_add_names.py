#!/usr/bin/env python3
"""Patch existing features_engineered.pkl to include `feature_names` key for downstream notebooks.
Usage: run on the machine where cooked_data is mounted.
"""
from pathlib import Path
import pickle
import sys

COOKED = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
P1 = COOKED / 'step_03_features' / 'features_engineered.pkl'
P2 = COOKED / 'features_engineered.pkl'

for p in (P1, P2):
    if p.exists():
        print('Patching', p)
        with open(p, 'rb') as f:
            data = pickle.load(f)

        # prefer 'feature_cols' or 'feature_cols' inside data
        feature_names = data.get('feature_cols') or data.get('feature_names')
        if feature_names is None and 'features_all' in data:
            try:
                feature_names = list(data['features_all'].columns)
            except Exception:
                feature_names = None

        if feature_names is None:
            print('No feature column list found in', p)
            continue

        data['feature_names'] = list(feature_names)

        # write back safely
        backup = p.with_suffix('.pkl.bak')
        p.rename(backup)
        with open(p, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        print('Patched and backed up original to', backup)
        sys.exit(0)

print('No features_engineered.pkl found at expected locations.')
sys.exit(2)
