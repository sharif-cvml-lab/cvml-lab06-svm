"""Dataset loading, embedding cache, model I/O and evaluation (fixed infrastructure).

No Qt in this file: the GUI (through workers.py) and the example scripts use the
very same functions.
"""

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np

from config import (
    EMBEDDING_CACHE_PATH,
    IMAGE_EXTENSIONS,
    SVM_MODEL_PATH,
    UNKNOWN_LABEL,
)
from svm import predict_svm, train_svm   # the student's functions


# --------------------------------------------------------------------------- #
# Dataset scanning / image reading
# --------------------------------------------------------------------------- #
def list_identities(directory):
    """{person name: [image paths]} for a directory with one sub-folder per person."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Folder not found: {directory}")
    identities = {}
    for person_dir in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not person_dir.is_dir():
            continue
        images = sorted(
            p for p in person_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            identities[person_dir.name] = images
    return identities


def read_image(path):
    """Read an image (unicode-safe on Windows). Returns BGR array or None."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Embeddings with a small on-disk cache
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingSet:
    embeddings: np.ndarray                      # (N, 512) float32
    labels: list                                # N folder names
    paths: list                                 # N image paths
    skipped: list = field(default_factory=list)  # [(path, reason)]
    cancelled: bool = False


def _load_cache(signature):
    try:
        with open(EMBEDDING_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache.get("signature") == signature:
            return cache["entries"]
    except (OSError, pickle.UnpicklingError, EOFError, KeyError, AttributeError):
        pass
    return {}


def _save_cache(signature, entries):
    try:
        EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump({"signature": signature, "entries": entries}, f)
    except OSError:
        pass   # the cache is only an optimisation


def compute_embeddings(extractor, directory, progress=None, should_cancel=None):
    """Embedding of the largest face of every image in `directory`.

    progress(done, total, message) and should_cancel() are optional callbacks.
    Images without a usable face are skipped and reported in `skipped`.
    """
    identities = list_identities(directory)
    total = sum(len(v) for v in identities.values())
    entries = _load_cache(extractor.signature)

    embeddings, labels, paths, skipped = [], [], [], []
    done = 0
    cancelled = False
    for person, images in identities.items():
        for path in images:
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            key = str(path.resolve())
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)

            cached = entries.get(key)
            if cached is not None and cached[0] == signature:
                embedding = cached[1]
            else:
                image = read_image(path)
                if image is None:
                    skipped.append((path, "unreadable image"))
                    done += 1
                    continue
                found = extractor.extract(image, retry_with_padding=True)
                embedding = None if found is None else found["identity_features"]
                entries[key] = (signature, embedding)

            if embedding is None:
                skipped.append((path, "no face found"))
            else:
                embeddings.append(embedding)
                labels.append(person)
                paths.append(path)
            done += 1
            if progress is not None:
                progress(done, total, f"{person}  ({done}/{total})")
        if cancelled:
            break

    _save_cache(extractor.signature, entries)
    matrix = np.array(embeddings, dtype=np.float32).reshape(-1, 512)
    return EmbeddingSet(matrix, labels, paths, skipped, cancelled)


# --------------------------------------------------------------------------- #
# Model validation, training, saving / loading, prediction
# --------------------------------------------------------------------------- #
def validate_model(model):
    """Return None if `model` follows the contract of svm.train_svm(), else a message."""
    if not isinstance(model, dict):
        return "train_svm() must return a dict"
    for key in ("classifier", "class_names", "centroids"):
        if key not in model:
            return f"the model dict has no '{key}' entry"
    if len(model["class_names"]) != len(model["centroids"]):
        return "'class_names' and 'centroids' must have the same length"
    return None


def model_class_names(model):
    return list(model["class_names"])


def train_model(extractor, database_dir, svm_params, progress=None, should_cancel=None):
    """Embed the whole database, call the student's train_svm() and save the model.

    Returns (model, info dict). Raises RuntimeError with a readable message.
    """
    data = compute_embeddings(extractor, database_dir, progress, should_cancel)
    if data.cancelled:
        raise RuntimeError("Training cancelled.")
    classes = sorted(set(data.labels))
    if len(classes) < 2:
        raise RuntimeError(
            f"Need at least 2 people with usable face images in '{database_dir}' "
            f"(found {len(classes)})."
        )

    started = time.perf_counter()
    model = train_svm(
        data.embeddings, list(data.labels),
        kernel=svm_params.kernel, C=svm_params.C, gamma=svm_params.gamma,
    )
    train_seconds = time.perf_counter() - started

    if model is None:
        raise NotImplementedError("train_svm() in svm.py is not implemented yet (it returned None).")
    problem = validate_model(model)
    if problem:
        raise RuntimeError(f"train_svm() problem: {problem}.")

    info = {
        "num_images": len(data.labels),
        "num_classes": len(model["class_names"]),
        "skipped": data.skipped,
        "train_seconds": train_seconds,
        "database_dir": str(database_dir),
        "kernel": svm_params.kernel, "C": svm_params.C, "gamma": svm_params.gamma,
    }
    save_model(model, info)
    return model, info


def save_model(model, info):
    SVM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in info.items() if k != "skipped"}
    joblib.dump({"model": model, "meta": meta}, SVM_MODEL_PATH)


def load_model():
    """Load assets/models/svm_model.joblib -> (model, meta), or (None, None)."""
    if not SVM_MODEL_PATH.is_file():
        return None, None
    try:
        bundle = joblib.load(SVM_MODEL_PATH)
        model, meta = bundle["model"], bundle.get("meta", {})
    except Exception:
        return None, None
    if validate_model(model) is not None:
        return None, None
    return model, meta


def predict_identity(model, embedding):
    """Call the student's predict_svm() and check its result -> (name, similarity)."""
    result = predict_svm(model, embedding)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError("predict_svm() must return a tuple (name, similarity)")
    name, similarity = result
    return str(name), float(similarity)


def decide(name, similarity, threshold):
    """Final decision shown to the user: the name if similarity >= threshold, else 'Unknown'."""
    return name if similarity >= threshold else UNKNOWN_LABEL


# --------------------------------------------------------------------------- #
# Evaluation on the test folder
# --------------------------------------------------------------------------- #
def evaluate_directory(extractor, model, test_dir, progress=None, should_cancel=None):
    """Predict every test image. Returns (records, skipped).

    record = {"path", "true", "pred", "score"}; pred/score are the *raw* SVM
    output (before thresholding), so the threshold can be changed afterwards.
    """
    data = compute_embeddings(extractor, test_dir, progress, should_cancel)
    if data.cancelled:
        raise RuntimeError("Evaluation cancelled.")
    records = []
    for embedding, true_label, path in zip(data.embeddings, data.labels, data.paths):
        name, score = predict_identity(model, embedding)
        records.append({"path": path, "true": true_label, "pred": name, "score": score})
    return records, data.skipped


def compute_metrics(records, known_classes, threshold):
    """Count outcomes at one threshold.

    known person  (true label was in the training set):
        correct match = accepted and right name | wrong match = accepted, wrong name
        false reject  = rejected as Unknown
    unseen person (true label was NOT in the training set):
        correct reject = rejected as Unknown    | false accept = accepted as someone
    """
    known = set(known_classes)
    m = dict(correct_match=0, wrong_match=0, false_reject=0,
             correct_reject=0, false_accept=0, n_known=0, n_unseen=0)
    for r in records:
        accepted = r["score"] >= threshold
        if r["true"] in known:
            m["n_known"] += 1
            if not accepted:
                m["false_reject"] += 1
            elif r["pred"] == r["true"]:
                m["correct_match"] += 1
            else:
                m["wrong_match"] += 1
        else:
            m["n_unseen"] += 1
            if accepted:
                m["false_accept"] += 1
            else:
                m["correct_reject"] += 1
    n = m["n_known"] + m["n_unseen"]
    m["n"] = n
    m["accuracy"] = (m["correct_match"] + m["correct_reject"]) / n if n else 0.0
    return m


def _pct(count, total):
    return f"{100.0 * count / total:5.1f}%" if total else "   n/a"


def format_report(records, known_classes, threshold, skipped=()):
    """Human-readable evaluation summary (threshold sweep included)."""
    m = compute_metrics(records, known_classes, threshold)
    lines = [
        f"Test images evaluated : {m['n']}   (no face found: {len(skipped)})",
        f"Threshold             : {threshold:.2f}",
        "",
        f"People that are in the training set  ({m['n_known']} images)",
        f"  correct match   {m['correct_match']:4d}  {_pct(m['correct_match'], m['n_known'])}",
        f"  wrong match     {m['wrong_match']:4d}  {_pct(m['wrong_match'], m['n_known'])}"
        "   (accepted, but another person's name)",
        f"  rejected        {m['false_reject']:4d}  {_pct(m['false_reject'], m['n_known'])}"
        "   (shown as Unknown)",
        "",
        f"People that are NOT in the training set  ({m['n_unseen']} images)",
        f"  correct reject  {m['correct_reject']:4d}  {_pct(m['correct_reject'], m['n_unseen'])}"
        "   (shown as Unknown)",
        f"  false accept    {m['false_accept']:4d}  {_pct(m['false_accept'], m['n_unseen'])}"
        "   (matched to a wrong person)",
        "",
        f"Overall accuracy (correct match + correct reject): {100.0 * m['accuracy']:.1f}%",
        "",
        "Threshold sweep",
        "  thr   correct  wrong  reject | correct  false  | accuracy",
        "        match    match  (known)| reject   accept |",
    ]
    best_thr, best_acc = 0.0, -1.0
    for step in range(0, 21):
        thr = step * 0.05
        s = compute_metrics(records, known_classes, thr)
        mark = ""
        if s["accuracy"] > best_acc + 1e-12:
            best_acc, best_thr = s["accuracy"], thr
        lines.append(
            f"  {thr:.2f}  {s['correct_match']:6d}  {s['wrong_match']:5d}  {s['false_reject']:6d} |"
            f" {s['correct_reject']:6d}  {s['false_accept']:6d}  | {100.0 * s['accuracy']:6.1f}%{mark}"
        )
    lines.append("")
    lines.append(f"Best overall accuracy in the sweep: {100.0 * best_acc:.1f}% at threshold {best_thr:.2f}")

    errors = [r for r in records
              if r["true"] in set(known_classes) and r["score"] >= threshold and r["pred"] != r["true"]]
    if errors:
        lines.append("")
        lines.append("Wrong matches at this threshold:")
        for r in errors[:15]:
            lines.append(f"  {r['true']}  ->  {r['pred']}  (sim {r['score']:.2f})")
    if skipped:
        lines.append("")
        lines.append("Images without a usable face:")
        for path, reason in list(skipped)[:15]:
            lines.append(f"  {Path(path).parent.name}/{Path(path).name}: {reason}")
    return "\n".join(lines)
