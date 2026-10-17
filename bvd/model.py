"""Temporal classifier on top of per-frame features."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import CFG


def context_features(Z: np.ndarray, k: int) -> np.ndarray:
    """Для каждого кадра: отклонение от медианы видео (убирает специфику камеры),
    среднее / std / max в окне ±k, и изменение 'будущее - прошлое' (движение, уход)."""
    T = len(Z)
    Zc = Z - np.median(Z, axis=0, keepdims=True)
    pad = np.pad(Zc, ((k, k), (0, 0)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1, axis=0)  # (T, D, 2k+1)
    fwd = pad[2 * k:2 * k + T] - pad[:T]
    return np.hstack([Zc, win.mean(-1), win.std(-1), win.max(-1), fwd]).astype(np.float32)


def smooth(p: np.ndarray, k: int = CFG.smooth_k) -> np.ndarray:
    if k <= 0 or len(p) < 2:
        return p
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(np.pad(p, k, mode="edge"), ker, mode="valid")


class ViolationModel:
    def __init__(self, n_pca=CFG.n_pca, k=CFG.context_k, C=CFG.C):
        self.n_pca, self.k, self.C = n_pca, k, C
        self.threshold = CFG.default_threshold
        self.type_clf = None
        self.type_classes: list[str] = []

    def _X(self, feat) -> np.ndarray:
        base = np.hstack([feat["low"], self.pca.transform(feat["clip"])])
        return context_features(base, self.k)

    def fit(self, feats: list[dict], labels: list):
        clip_all = np.vstack([f["clip"] for f in feats])
        self.pca = PCA(n_components=min(self.n_pca, clip_all.shape[0] - 1), random_state=0).fit(clip_all)

        X, y, w, Xt, yt = [], [], [], [], []
        for f, lab in zip(feats, labels):
            C = self._X(f)
            t = f["times"]
            if lab.violation and lab.ts is not None:
                d = np.abs(t - lab.ts)
                pos, neg = d <= CFG.pos_radius, d > CFG.ignore_radius
                if pos.sum() == 0:  # ts за пределами видео — берём ближайший кадр
                    pos = d == d.min()
                X += [C[pos], C[neg]]
                y += [np.ones(pos.sum()), np.zeros(neg.sum())]
                w += [np.ones(pos.sum()), np.full(neg.sum(), CFG.neg_weight_in_pos_video)]
                if lab.vtype:
                    Xt.append(C[pos]); yt += [lab.vtype] * int(pos.sum())
            else:
                X.append(C); y.append(np.zeros(len(C))); w.append(np.ones(len(C)))
        X, y, w = np.vstack(X), np.concatenate(y), np.concatenate(w)
        # балансировка классов вручную (совместно с весами кадров)
        w = w * np.where(y == 1, w[y == 0].sum() / max(w[y == 1].sum(), 1e-6), 1.0)

        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=self.C, max_iter=5000).fit(self.scaler.transform(X), y, sample_weight=w)

        if Xt and len(set(yt)) >= 2:
            Xt = self.scaler.transform(np.vstack(Xt))
            self.type_clf = LogisticRegression(C=self.C, max_iter=5000, class_weight="balanced").fit(Xt, yt)
            self.type_classes = list(self.type_clf.classes_)
        return self

    def frame_probs(self, feat) -> np.ndarray:
        Xs = self.scaler.transform(self._X(feat))
        return smooth(self.clf.predict_proba(Xs)[:, 1])

    def predict(self, feat) -> dict:
        Xs = self.scaler.transform(self._X(feat))
        probs = smooth(self.clf.predict_proba(Xs)[:, 1])
        i = int(np.argmax(probs))
        res = {"probs": probs, "times": feat["times"], "score": float(probs[i]),
               "peak_idx": i, "peak_time": float(feat["times"][i]),
               "violation": bool(probs[i] >= self.threshold), "type": None, "type_probs": {}}
        if self.type_clf is not None:
            lo, hi = max(0, i - self.k), min(len(Xs), i + self.k + 1)
            tp = self.type_clf.predict_proba(Xs[lo:hi]).mean(0)
            res["type_probs"] = {c: float(p) for c, p in zip(self.type_classes, tp)}
            res["type"] = self.type_classes[int(np.argmax(tp))]
        return res
