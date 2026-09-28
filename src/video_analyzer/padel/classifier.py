"""Stroke classifier trained on annotated hits (pose sequence + court context).

Features (FEATURE_VERSION 1), per hit:
- 7 keypoints (racket wrist/elbow/shoulder, other wrist, nose, hip centre, ankle centre) over
  ±15 frames (±0.5 s) sampled every 3 frames → 11 steps × 7 × 2 = 154 values,
  centred on the hip centre at impact and scaled by body height at impact;
- canonical view: far players (facing the camera) are mirrored to look "seen from behind" like
  near players, left-handers mirrored so the racket arm is always the right one;
- context: distance to the net, lateral position in own half, contact height (one-hot), wrist
  speed (raw and noise-floor normalised).
Model: small MLP (PyTorch, CPU is enough), strong weight decay, noise/time-shift augmentation.
Metrics come from stratified k-fold cross-validation over all labelled videos.
"""
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .players import POSE_FPS, ShotTracks
from .strokes import Hit

FEATURE_VERSION = 1
WINDOW, STEP = 15, 3
MIN_PER_CLASS = 4
MERGE = {"bandeja": "bandeja_vibora", "vibora": "bandeja_vibora", "bandeja_ou_vibora": "bandeja_vibora"}
IGNORE = {"incertain"}
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def target(label: str) -> str | None:
    if label in IGNORE:
        return None
    return MERGE.get(label, label)


def _interp_missing(xy: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """xy (T, K, 2), ok (T, K): linear interpolation over time of low-confidence points."""
    t = np.arange(len(xy))
    for k in range(xy.shape[1]):
        good = ok[:, k]
        if good.sum() >= 2:
            for c in (0, 1):
                xy[:, k, c] = np.interp(t, t[good], xy[good, k, c])
        elif good.sum() == 1:
            xy[:, k] = xy[good, k][0]
        else:
            xy[:, k] = np.nan
    return xy


def hit_features(h: Hit, tr: ShotTracks) -> np.ndarray | None:
    T = len(tr.kpts)
    idx = np.clip(np.arange(h.frame - WINDOW, h.frame + WINDOW + 1), 0, T - 1)
    k = tr.kpts[idx, h.slot].copy()  # (31, 17, 3)
    ok = k[..., 2] > 0.3
    if ok[WINDOW].sum() < 6:
        return None
    xy = _interp_missing(k[..., :2].astype(np.float64), ok)
    if h.slot < 2:            # far side faces the camera: mirror to a "from behind" view
        xy[..., 0] *= -1
    if h.arm == "gauche":     # left-hander: mirror + swap anatomical sides
        xy[..., 0] *= -1
        for a, b in LR_PAIRS:
            xy[:, [a, b]] = xy[:, [b, a]]
    hips = xy[:, [11, 12]].mean(axis=1)
    ankles = xy[:, [15, 16]].mean(axis=1)
    c = hips[WINDOW]
    if np.isnan(c).any():
        return None
    head = xy[WINDOW, 0] if not np.isnan(xy[WINDOW, 0]).any() else xy[WINDOW, [5, 6]].mean(axis=0)
    scale = np.linalg.norm(head - ankles[WINDOW]) if not np.isnan(ankles[WINDOW]).any() else \
        2.2 * np.linalg.norm(head - c)
    if not np.isfinite(scale) or scale < 10:
        return None
    pts = np.stack([xy[:, 10], xy[:, 8], xy[:, 6], xy[:, 9], xy[:, 0], hips, ankles], axis=1)  # (31, 7, 2)
    pts = (pts - c) / scale
    pts = np.nan_to_num(pts[::STEP], nan=0.0).ravel()
    lateral = (10 - h.x) if h.slot < 2 else h.x
    ctx = [h.dist_net / 10, lateral / 10, float(h.height == "bas"), float(h.height == "haut"),
           float(h.height == "tete"), min(h.speed, 2.0), min(h.evidence, 20) / 10]
    return np.concatenate([pts, ctx]).astype(np.float32)


@dataclass
class Sample:
    x: np.ndarray
    y: str
    video: str
    key: str
    vlm: str
    pose: str


def _mlp(n_in: int, n_out: int):
    import torch.nn as nn

    return nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_out))


def _fit(X: np.ndarray, y: np.ndarray, n_cls: int, epochs: int = 300, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y)
    counts = np.bincount(y, minlength=n_cls).astype(np.float32)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1) / n_cls)
    model = _mlp(X.shape[1], n_cls)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    for _ in range(epochs):
        model.train()
        noise = torch.tensor(rng.normal(0, 0.15, Xn.shape), dtype=torch.float32)
        opt.zero_grad()
        loss = lossf(model(Xn + noise), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model, mu, sd


def _predict(model, mu, sd, X: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return torch.softmax(model(torch.tensor((X - mu) / sd, dtype=torch.float32)), dim=1).numpy()


def _folds(y: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    """Stratified fold assignment."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return [np.where(fold == i)[0] for i in range(k)]


def train(samples: list[Sample], k: int = 5) -> tuple[dict, dict]:
    """Returns (model bundle, report). Classes with < MIN_PER_CLASS examples are left out."""
    counts = Counter(s.y for s in samples)
    classes = sorted(c for c, n in counts.items() if n >= MIN_PER_CLASS)
    used = [s for s in samples if s.y in classes]
    if len(classes) < 2 or len(used) < 10:
        raise ValueError(f"pas assez d'annotations ({len(used)} exemples, classes {dict(counts)})")
    X = np.stack([s.x for s in used])
    y = np.array([classes.index(s.y) for s in used])
    k = max(2, min(k, min(np.bincount(y))))
    oof = np.zeros((len(y), len(classes)))
    for test in _folds(y, k):
        train_idx = np.setdiff1d(np.arange(len(y)), test)
        m, mu, sd = _fit(X[train_idx], y[train_idx], len(classes))
        oof[test] = _predict(m, mu, sd, X[test])
    pred = oof.argmax(axis=1)
    cv_pred = [classes[i] for i in pred]
    truth = [s.y for s in used]
    vlm = [target(s.vlm) if s.vlm else None for s in used]
    pose = [target(s.pose) if s.pose else None for s in used]

    def acc(p):
        pairs = [(a, b) for a, b in zip(p, truth) if a is not None]
        return (sum(a == b for a, b in pairs) / len(pairs), len(pairs)) if pairs else (float("nan"), 0)

    report = {
        "n": len(used), "classes": classes, "counts": {c: counts[c] for c in classes},
        "left_out": {c: n for c, n in counts.items() if c not in classes}, "folds": k,
        "accuracy": {"classifieur_cv": acc(cv_pred), "vlm": acc(vlm), "pose": acc(pose)},
        "confusion_classifier": confusion(truth, cv_pred, classes),
        "cv_predictions": {s.key: p for s, p in zip(used, cv_pred)},
    }
    model, mu, sd = _fit(X, y, len(classes))
    bundle = {"model": model, "mu": mu, "sd": sd, "classes": classes, "feature_version": FEATURE_VERSION,
              "cv_accuracy": report["accuracy"]["classifieur_cv"][0], "vlm_accuracy": report["accuracy"]["vlm"][0],
              "n": len(used), "pose_fps": POSE_FPS}  # features are per frame: tracks must be sampled at this rate
    return bundle, report


def confusion(truth: list[str], pred: list[str], classes: list[str]) -> dict[str, dict[str, int]]:
    m = {c: dict.fromkeys(classes, 0) for c in classes}
    for t, p in zip(truth, pred):
        if t in m and p in m[t]:
            m[t][p] += 1
    return m


def save(bundle: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": bundle["model"].state_dict(), "n_in": len(bundle["mu"]),
                **{k: v for k, v in bundle.items() if k != "model"}}, path)
    meta = {k: v for k, v in bundle.items() if k not in ("model", "mu", "sd")}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    import torch

    d = torch.load(path, weights_only=False)
    if d.get("feature_version") != FEATURE_VERSION:
        return None
    model = _mlp(d["n_in"], len(d["classes"]))
    model.load_state_dict(d["state"])
    model.eval()
    return {**d, "model": model}


def predict(bundle: dict, h: Hit, tr: ShotTracks) -> tuple[str, float] | None:
    x = hit_features(h, tr)
    if x is None:
        return None
    p = _predict(bundle["model"], bundle["mu"], bundle["sd"], x[None])[0]
    i = int(p.argmax())
    return bundle["classes"][i], float(p[i])
