"""Phase 6: measure clustering accuracy on LFW against the true identities."""
import argparse
import sqlite3
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save the chart to a file, no window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import pair_confusion_matrix

from facevault.cluster import cluster_faces, cluster_faces_hdbscan

DB_PATH = "facevault.db"
DATASET_DIR = "faces_db"
CHART_PATH = Path("docs/eps_tuning.png")


def load_labeled(db_path, dataset_dir):
    """Largest face of each LFW photo, with the true name from its folder."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """
        SELECT f.photo_id, f.embedding, p.path, (f.x2 - f.x1) * (f.y2 - f.y1) AS area
        FROM faces f JOIN photos p ON p.id = f.photo_id
        WHERE p.path LIKE ?
        ORDER BY f.photo_id, area DESC
        """,
        (f"%{dataset_dir}%",),
    ).fetchall()
    con.close()

    seen, names, embs = set(), [], []
    for photo_id, emb, path, _ in rows:
        if photo_id in seen:
            continue  # keep only the largest face per photo
        seen.add(photo_id)
        names.append(Path(path).parent.name)
        embs.append(np.frombuffer(emb, dtype=np.float32))
    if not embs:
        raise SystemExit(f"No faces found from photos under '{dataset_dir}'.")

    X = np.vstack(embs)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    _, y_true = np.unique(names, return_inverse=True)
    return X, y_true


def score(y_true, labels):
    """ARI plus pairwise precision / recall / F1. Noise faces count as alone."""
    pred = labels.copy()
    noise = pred == -1
    pred[noise] = np.arange(noise.sum()) + pred.max() + 1  # each noise face its own group

    C = pair_confusion_matrix(y_true, pred)
    tp, fp, fn = C[1, 1], C[0, 1], C[1, 0]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return {
        "ari": adjusted_rand_score(y_true, pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clusters": n_clusters,
        "noise": int(noise.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate clustering on LFW")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--min-samples", type=int, default=3)
    args = parser.parse_args()

    X, y_true = load_labeled(args.db, args.dataset)
    print(f"{len(X)} labeled faces, {len(set(y_true))} true identities\n")

    eps_values = np.round(np.arange(0.30, 0.701, 0.025), 3)
    results = []
    header = f"{'eps':>6} {'clusters':>8} {'noise':>6} {'ARI':>6} {'prec':>6} {'recall':>6} {'F1':>6}"
    print(header)
    for eps in eps_values:
        r = score(y_true, cluster_faces(X, eps=eps, min_samples=args.min_samples))
        results.append(r)
        print(f"{eps:>6.3f} {r['clusters']:>8} {r['noise']:>6} {r['ari']:>6.3f} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")

    best = int(np.argmax([r["f1"] for r in results]))
    print(f"\nBest eps by pairwise F1: {eps_values[best]:.3f}")

    # Compare against HDBSCAN and time both at the chosen setting
    t0 = time.perf_counter()
    db_best = score(y_true, cluster_faces(X, eps=eps_values[best], min_samples=args.min_samples))
    t_db = time.perf_counter() - t0
    t0 = time.perf_counter()
    hdb = score(y_true, cluster_faces_hdbscan(X, min_cluster_size=args.min_samples))
    t_hdb = time.perf_counter() - t0

    print(f"\n{'method':<20} {'clusters':>8} {'noise':>6} {'ARI':>6} {'prec':>6} {'recall':>6} {'F1':>6} {'time':>7}")
    for name, r, t in [(f"DBSCAN eps={eps_values[best]:.3f}", db_best, t_db), ("HDBSCAN", hdb, t_hdb)]:
        print(f"{name:<20} {r['clusters']:>8} {r['noise']:>6} {r['ari']:>6.3f} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {t:>6.2f}s")

    # Chart
    CHART_PATH.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in [("precision", "Pairwise precision"), ("recall", "Pairwise recall"),
                       ("f1", "Pairwise F1"), ("ari", "Adjusted Rand Index")]:
        ax.plot(eps_values, [r[key] for r in results], marker="o", markersize=3, label=label)
    ax.axvline(eps_values[best], color="grey", linestyle="--", linewidth=1)
    ax.annotate(f"best eps = {eps_values[best]:.3f}", (eps_values[best], 0.02),
                xytext=(5, 0), textcoords="offset points", fontsize=9, color="grey")
    ax.set_xlabel("DBSCAN eps (cosine distance)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Clustering quality vs eps on LFW ({len(X)} faces, {len(set(y_true))} people)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)
    print(f"\nChart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()