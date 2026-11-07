"""Your personal gesture model.

Stores recorded hand *landmarks* (21 points per sample, no images) and
recognises poses with a k-nearest-neighbour vote. Poses that look unlike
anything recorded are rejected, so the built-in rules still apply.
"""
import json
import math
from pathlib import Path

import numpy as np

from .config import ROOT

MODEL_FILE = ROOT / "models" / "my_gestures.npz"

# Poses decided by the pinch-distance rules (they need hysteresis for reliable
# clicks), so they are not trained.
RULE_ONLY = {"pinch", "pinch_middle", "ok", "none"}

TIPS = (4, 8, 12, 16, 20)


def features(lm):
    """Translation/scale-invariant features. Orientation is kept on purpose,
    otherwise thumbs-up and thumbs-down would look identical."""
    p = np.asarray(lm, dtype=np.float32)[:, :2]
    p = p - p[0]
    size = float(np.linalg.norm(p[9])) or 1e-6
    p = p / size
    tips = p[list(TIPS)]
    d = np.linalg.norm(tips[:, None, :] - tips[None, :, :], axis=-1)
    iu = np.triu_indices(5, 1)
    # fingertip-to-fingertip distances weigh more: they describe the pose shape
    return np.concatenate([p[1:].ravel(), 1.5 * d[iu]]).astype(np.float32)


class GestureModel:
    def __init__(self, X=None, y=None, k=5):
        self.X = np.zeros((0, 50), np.float32) if X is None else np.asarray(X, np.float32)
        self.y = np.array([], dtype=object) if y is None else np.asarray(y, dtype=object)
        self.k = k
        self.threshold = None
        if len(self.y):
            self.fit()

    # ---------------------------------------------------------------- data
    @property
    def labels(self):
        return sorted(set(self.y.tolist()))

    def counts(self):
        return {l: int((self.y == l).sum()) for l in self.labels}

    def add(self, label, feats):
        feats = np.asarray(feats, np.float32).reshape(-1, self.X.shape[1] if len(self.X) else 50)
        self.X = np.vstack([self.X, feats])
        self.y = np.concatenate([self.y, np.array([label] * len(feats), dtype=object)])

    def remove(self, label):
        keep = self.y != label
        self.X, self.y = self.X[keep], self.y[keep]

    # ------------------------------------------------------------ training
    def fit(self):
        """Pick a rejection distance from how spread out each class is."""
        if len(self.y) < 2:
            self.threshold = None
            return
        within = []
        for label in self.labels:
            Xi = self.X[self.y == label]
            if len(Xi) < 2:
                continue
            # distance to the k-th nearest same-class neighbour
            D = np.linalg.norm(Xi[:, None] - Xi[None], axis=-1)
            D.sort(axis=1)
            kk = min(self.k, len(Xi) - 1)
            within.append(D[:, kk])
        if not within:
            self.threshold = None
            return
        w = np.concatenate(within)
        self.threshold = float(np.percentile(w, 95) * 2.0 + 0.05)

    def predict(self, lm):
        """Return (label, confidence 0..1). label is None if unsure."""
        if len(self.y) == 0:
            return None, 0.0
        f = features(lm)
        d = np.linalg.norm(self.X - f, axis=1)
        k = min(self.k, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        if self.threshold is not None and float(d[idx].mean()) > self.threshold:
            return None, 0.0
        votes = {}
        for i in idx:  # distance-weighted vote
            votes[self.y[i]] = votes.get(self.y[i], 0.0) + 1.0 / (d[i] + 1e-3)
        label = max(votes, key=votes.get)
        return label, votes[label] / sum(votes.values())

    def evaluate(self, folds=5, seed=0):
        """Cross-validated accuracy per label (blocks of consecutive samples
        are held out together, so it is not fooled by near-duplicate frames)."""
        n = len(self.y)
        if n < folds * 2 or len(self.labels) < 2:
            return None
        # Split every pose's samples into contiguous blocks, one block per fold.
        fold_of = np.zeros(n, dtype=int)
        rng = np.random.default_rng(seed)
        for label in self.labels:
            idx = np.flatnonzero(self.y == label)
            fold_of[idx] = rng.permutation(folds)[(np.arange(len(idx)) * folds) // len(idx)]
        correct = {l: 0 for l in self.labels}
        total = {l: 0 for l in self.labels}
        confusions = {}
        for f in range(folds):
            tr, te = fold_of != f, fold_of == f
            if len(set(self.y[tr].tolist())) < 2:
                continue
            m = GestureModel(self.X[tr], self.y[tr], self.k)
            m.threshold = None  # measure the vote only
            for xf, yt in zip(self.X[te], self.y[te]):
                d = np.linalg.norm(m.X - xf, axis=1)
                kk = min(m.k, len(d))
                idx = np.argpartition(d, kk - 1)[:kk]
                votes = {}
                for i in idx:
                    votes[m.y[i]] = votes.get(m.y[i], 0.0) + 1.0 / (d[i] + 1e-3)
                yp = max(votes, key=votes.get)
                total[yt] += 1
                correct[yt] += int(yp == yt)
                if yp != yt:
                    confusions[(yt, yp)] = confusions.get((yt, yp), 0) + 1
        acc = {l: (correct[l] / total[l] if total[l] else math.nan) for l in self.labels}
        return acc, confusions

    # ------------------------------------------------------------- storage
    def save(self, path: Path = MODEL_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, y=np.array(self.y.tolist(), dtype=str),
                            meta=json.dumps({"k": self.k, "version": 1}))

    @classmethod
    def load(cls, path: Path = MODEL_FILE):
        path = Path(path)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        return cls(data["X"], data["y"].astype(object), k=meta.get("k", 5))
