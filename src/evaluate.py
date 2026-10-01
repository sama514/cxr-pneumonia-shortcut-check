import numpy as np
from sklearn.metrics import roc_auc_score


def youden_threshold(y, score):
    best_t, best_j = 0.5, -1
    for t in np.unique(score):
        pred = score >= t
        tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def metrics(y, score, thr=0.5):
    y = np.asarray(y); pred = (np.asarray(score) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "auc": float(roc_auc_score(y, score)),
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp, "threshold": float(thr),
    }


def bootstrap_auc_ci(y, score, groups, n=1000, seed=0):
    """95% CI for AUC, resampling PATIENTS (not images)."""
    y = np.asarray(y); score = np.asarray(score); groups = np.asarray(groups)
    idx_by = {}
    for i, g in enumerate(groups):
        idx_by.setdefault(g, []).append(i)
    keys = list(idx_by); rng = np.random.default_rng(seed); aucs = []
    for _ in range(n):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([idx_by[keys[k]] for k in pick])
        if len(set(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
