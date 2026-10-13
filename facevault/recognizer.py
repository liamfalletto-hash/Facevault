"""Phase 9: recognise named people on camera tracks by voting across frames."""
import sqlite3
from collections import Counter

import numpy as np

UNKNOWN = "Unknown"


def _normalize(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)


def load_centroids(db_path):
    """One centroid per named person: mean of their normalised embeddings, re-normalised."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT p.name, f.embedding FROM faces f JOIN persons p ON p.id = f.person_id"
    ).fetchall()
    con.close()

    groups = {}
    for name, emb in rows:
        groups.setdefault(name, []).append(np.frombuffer(emb, dtype=np.float32))
    names = sorted(groups)
    if not names:
        return [], np.empty((0, 512), dtype=np.float32)
    centroids = np.vstack([_normalize(_normalize(np.vstack(groups[n])).mean(axis=0))
                           for n in names])
    return names, centroids


class Recognizer:
    def __init__(self, names, centroids, threshold=0.45, min_frames=3):
        self.names = names
        self.centroids = centroids
        self.threshold = threshold
        self.min_frames = min_frames

    @classmethod
    def from_db(cls, db_path, **kwargs):
        names, centroids = load_centroids(db_path)
        return cls(names, centroids, **kwargs)

    def label(self, embeddings):
        """Vote over a track's recent embeddings. Returns (name, similarity)."""
        embeddings = list(embeddings)
        if not self.names or not embeddings:
            return UNKNOWN, 0.0

        E = _normalize(np.vstack(embeddings))
        sims = E @ self.centroids.T          # cosine similarity, shape (frames, people)
        best = sims.argmax(axis=1)
        best_sim = sims.max(axis=1)

        # Each frame votes for its closest person, or Unknown if too far from everyone
        votes = [self.names[i] if s >= self.threshold else UNKNOWN
                 for i, s in zip(best, best_sim)]
        winner, count = Counter(votes).most_common(1)[0]

        enough_frames = len(votes) >= self.min_frames
        majority = count > len(votes) / 2
        if winner == UNKNOWN or not enough_frames or not majority:
            return UNKNOWN, float(best_sim.mean())

        winner_sims = [s for v, s in zip(votes, best_sim) if v == winner]
        return winner, float(np.mean(winner_sims))