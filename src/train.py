"""python -m src.train --data data"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data, features
from .evaluate import bootstrap_auc_ci, metrics, youden_threshold


def get_embeddings(paths, name, cache_dir):
    cache = Path(cache_dir) / f"emb_{name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["paths"]) == list(paths):
            print(f"Loaded cached embeddings from {cache}")
            return z["X"]
    X = features.EMBEDDERS[name](paths)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, paths=np.array(paths))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedder", default="resnet18", choices=list(features.EMBEDDERS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images")
    a = ap.parse_args()

    records = data.scan(a.data)
    if not records:
        raise SystemExit(f"No images found under '{a.data}'. Expected folders named NORMAL and PNEUMONIA somewhere inside.")
    if a.limit:
        import random
        random.Random(a.seed).shuffle(records)
        records = records[:a.limit]
    print(f"Found {len(records)} images, {len({r['patient'] for r in records})} patients")

    splits = data.split_by_patient(records, seed=a.seed)
    data.assert_no_patient_overlap(splits)
    print("OK: no patient appears in two splits")

    ordered = splits["train"] + splits["val"] + splits["test"]
    paths = [r["path"] for r in ordered]
    X_all = get_embeddings(paths, a.embedder, a.cache)
    S_all = features.simple_features(paths)
    n_tr, n_va = len(splits["train"]), len(splits["val"])
    sl = {"train": slice(0, n_tr), "val": slice(n_tr, n_tr + n_va), "test": slice(n_tr + n_va, None)}
    y = {k: np.array([r["label"] for r in splits[k]]) for k in sl}
    g = {k: np.array([r["patient"] for r in splits[k]]) for k in sl}
    for k in sl:
        if len(set(y[k])) < 2:
            raise SystemExit(f"Split '{k}' has only one class; dataset too small.")

    # Main model: choose C on validation AUC
    from sklearn.metrics import roc_auc_score
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
        m.fit(X_all[sl["train"]], y["train"])
        auc = roc_auc_score(y["val"], m.predict_proba(X_all[sl["val"]])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, m)
    val_auc, C, model = best
    p_val = model.predict_proba(X_all[sl["val"]])[:, 1]
    p_test = model.predict_proba(X_all[sl["test"]])[:, 1]
    thr = youden_threshold(y["val"], p_val)
    main_05 = metrics(y["test"], p_test, 0.5)
    main_tuned = metrics(y["test"], p_test, thr)
    ci = bootstrap_auc_ci(y["test"], p_test, g["test"])

    # Shortcut check: size + brightness only
    sc = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=a.seed)
    sc.fit(S_all[sl["train"]], y["train"])
    p_sc = sc.predict_proba(S_all[sl["test"]])[:, 1]
    sc_m = metrics(y["test"], p_sc, 0.5)
    sc_ci = bootstrap_auc_ci(y["test"], p_sc, g["test"])

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    result = {"embedder": a.embedder, "C": C, "val_auc": val_auc, "main_thr_0.5": main_05,
              "main_thr_val": main_tuned, "main_auc_ci95": ci, "shortcut": sc_m, "shortcut_auc_ci95": sc_ci,
              "counts": {k: dict(Counter(y[k].tolist())) for k in sl},
              "patients": {k: len(set(g[k])) for k in sl}}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    def row(name, m):
        return f"| {name} | {m['auc']:.3f} | {m['accuracy']:.3f} | {m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['tp']}/{m['fn']}/{m['tn']}/{m['fp']} |"

    md = [
        "# Results: chest X-ray NORMAL vs PNEUMONIA", "",
        f"Embedder: `{a.embedder}` | seed: {a.seed} | chosen C (by validation AUC): {C} | validation AUC: {val_auc:.3f}", "",
        "## Split sizes (images / patients)", "",
        "| Split | NORMAL | PNEUMONIA | Patients |", "|---|---|---|---|",
    ]
    for k in sl:
        c = Counter(y[k].tolist())
        md.append(f"| {k} | {c.get(0, 0)} | {c.get(1, 0)} | {len(set(g[k]))} |")
    md += ["", "## Test-set performance", "",
           "| Model | AUC | Accuracy | Sensitivity | Specificity | TP/FN/TN/FP |", "|---|---|---|---|---|---|",
           row("Main model (threshold 0.5)", main_05),
           row(f"Main model (threshold {thr:.3f} chosen on validation)", main_tuned),
           row("Shortcut model (size + brightness only)", sc_m), "",
           f"Main model AUC 95% CI (patient-level bootstrap): {ci[0]:.3f} to {ci[1]:.3f}",
           f"Shortcut model AUC 95% CI: {sc_ci[0]:.3f} to {sc_ci[1]:.3f}", "",
           "## How to read the shortcut check", "",
           "The shortcut model never sees lung content, only image size and overall brightness.",
           "- AUC near 0.5: no obvious shortcut from these simple properties.",
           "- AUC clearly above 0.5: the dataset leaks the label through acquisition properties, so part of the main model's score may not come from the lungs.",
           "- Compare it with the main model AUC. If they are close, be suspicious of the main result.", ""]
    (out / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved {out / 'metrics.md'}")


if __name__ == "__main__":
    main()
