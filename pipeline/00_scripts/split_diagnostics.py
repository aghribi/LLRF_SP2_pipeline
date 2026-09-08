"""
Per-category train/test composition logging for SPIRAL2 anomalies_exploration.

Why this exists: a 2026-09-05 audit found that no script anywhere in this
pipeline logs how many examples of each fault category actually land in the
train vs. test fold. With 7 categories ranging from 11 to 591 total events,
that's not a cosmetic gap -- a report claiming e.g. "F1=0.85 for Seuil
pick-up" (12 total events) is currently unverifiable: nobody can tell whether
the test fold got 2 examples or 6 of that category, and F1 on n=2 is not a
meaningful number regardless of what it reads. This module makes that
composition visible every time a script splits the data, so it's never
silently invisible again.

Usage, right after a leakage_safe_split(...) call:

    from split_diagnostics import log_split_composition
    split = leakage_safe_split(pkl_data, y, test_size=0.3, random_state=42)
    log_split_composition(split['y_train'], split['y_test'], label_names, logger)
"""

import numpy as np


def log_split_composition(y_train, y_test, label_names, logger, title="Split composition"):
    """
    Log (and return) a per-category train/test/total breakdown.

    y_train, y_test : either 1D integer-class arrays (one label per sample,
                       e.g. root-cause's argmax-derived y_root) or 2D binary
                       arrays (n_samples, n_labels) (multi-label, one column
                       per fault category). Both shapes are used across this
                       pipeline's scripts, so both are handled here rather
                       than forcing every caller to reshape first.
    label_names     : names in the same order as the class indices / columns.
    logger          : a stdlib logging.Logger (or anything with .info()).
    title           : header line, so the log clearly ties this table to the
                       specific script/model it came from.

    Returns a list of dicts (one per category): name, total, train, test,
    test_frac -- the same numbers that got logged, for any caller that wants
    to feed them into the results-manifest or a downstream diagnostic.
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    multilabel = y_train.ndim == 2

    rows = []
    for i, name in enumerate(label_names):
        if multilabel:
            train_n = int(np.sum(y_train[:, i] == 1))
            test_n = int(np.sum(y_test[:, i] == 1))
        else:
            train_n = int(np.sum(y_train == i))
            test_n = int(np.sum(y_test == i))
        total = train_n + test_n
        rows.append({
            'label': name,
            'total': total,
            'train': train_n,
            'test': test_n,
            'test_frac': (test_n / total) if total else float('nan'),
        })

    logger.info("=" * 78)
    logger.info(title)
    logger.info("=" * 78)
    logger.info(f"{'Category':<38} {'Total':>7} {'Train':>7} {'Test':>7} {'TestFrac':>9}")
    for r in rows:
        frac_str = f"{r['test_frac']:.1%}" if total_is_finite(r['test_frac']) else "n/a"
        logger.info(f"{r['label']:<38} {r['total']:>7} {r['train']:>7} {r['test']:>7} {frac_str:>9}")
        if r['test'] < 5:
            logger.warning(f"  ^ '{r['label']}': only {r['test']} test example(s) -- any "
                            f"per-category metric for this label is not statistically meaningful")
    logger.info("=" * 78)

    return rows


def total_is_finite(x):
    return x == x  # NaN != NaN; avoids importing math/np just for isnan here
