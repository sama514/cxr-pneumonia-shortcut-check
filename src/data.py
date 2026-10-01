"""Scan the dataset folder, extract patient ids, split by patient."""
import random
import re
from collections import defaultdict
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png"}


def label_from_path(p: Path):
    parts = [x.upper() for x in p.parts]
    if "PNEUMONIA" in parts:
        return 1
    if "NORMAL" in parts:
        return 0
    return None


def patient_id(filename: str, label: int) -> str:
    """Kaggle names: person12_bacteria_45.jpeg (pneumonia), IM-0115-0001.jpeg / NORMAL2-IM-1427-0001.jpeg (normal)."""
    stem = Path(filename).stem
    m = re.search(r"person(\d+)", stem, re.I)
    if m:
        return f"P{m.group(1)}"
    m = re.search(r"IM-(\d+)", stem, re.I)
    if m:
        return f"N{m.group(1)}"
    return f"F{label}_{stem}"  # unknown pattern: treat each file as its own patient


def scan(root):
    root = Path(root)
    seen, records = set(), []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXTS or "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        y = label_from_path(p)
        if y is None:
            continue
        key = (p.name, y)  # the Kaggle zip often has a duplicated nested folder
        if key in seen:
            continue
        seen.add(key)
        records.append({"path": str(p), "label": y, "patient": patient_id(p.name, y)})
    return records


def split_by_patient(records, seed=42, fractions=(0.70, 0.15, 0.15)):
    """Returns dict name -> list of records. Patients (not images) are shuffled and assigned per class."""
    by_patient = defaultdict(list)
    for r in records:
        by_patient[r["patient"]].append(r)
    by_label = defaultdict(list)
    for pid, rs in by_patient.items():
        majority = round(sum(r["label"] for r in rs) / len(rs))
        by_label[majority].append(pid)
    rng = random.Random(seed)
    assign = {}
    for label, pids in by_label.items():
        pids = sorted(pids)
        rng.shuffle(pids)
        n = len(pids)
        n_tr, n_va = int(n * fractions[0]), int(n * fractions[1])
        for i, pid in enumerate(pids):
            assign[pid] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    out = {"train": [], "val": [], "test": []}
    for pid, rs in by_patient.items():
        out[assign[pid]].extend(rs)
    return out


def assert_no_patient_overlap(splits):
    sets = {k: {r["patient"] for r in v} for k, v in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        both = sets[a] & sets[b]
        assert not both, f"Patient leakage between {a} and {b}: {sorted(both)[:5]}"
