"""Phase 4: cluster face embeddings into people."""
import argparse
import shutil
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_rand_score

DB_PATH = "facevault.db"
DEBUG_DIR = Path("debug_clusters")


# ---------- database ----------

def ensure_cluster_column(con):
    """Add faces.cluster_id if it doesn't exist yet."""
    cols = [row[1] for row in con.execute("PRAGMA table_info(faces)")]
    if "cluster_id" not in cols:
        con.execute("ALTER TABLE faces ADD COLUMN cluster_id INTEGER")
        con.commit()


def load_embeddings(con):
    """Return (ids, X) where X has one normalised embedding per row."""
    rows = con.execute(
        "SELECT id, embedding FROM faces WHERE embedding IS NOT NULL"  # ADJUST names
    ).fetchall()
    if not rows:
        raise SystemExit("No embeddings found. Run the Phase 3 scan first.")
    ids = [r[0] for r in rows]
    X = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])  # ADJUST dtype
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return ids, X


def save_labels(con, ids, labels):
    con.executemany(
        "UPDATE faces SET cluster_id = ? WHERE id = ?",
        [(int(label), face_id) for face_id, label in zip(ids, labels)],
    )
    con.commit()


# ---------- clustering ----------

def cluster_faces(X, eps=0.5, min_samples=3):
    """DBSCAN. Label -1 means noise (a face that fits no group)."""
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(X)


def cluster_faces_hdbscan(X, min_cluster_size=3):
    """HDBSCAN. X is normalised, so euclidean distance behaves like cosine."""
    return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)


def summarize(labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    return n_clusters, n_noise


# ---------- commands ----------

def cmd_run(con, args):
    ids, X = load_embeddings(con)
    print(f"Loaded {X.shape[0]} faces, {X.shape[1]} dimensions each")
    if args.method == "hdbscan":
        labels = cluster_faces_hdbscan(X, min_cluster_size=args.min_samples)
    else:
        labels = cluster_faces(X, eps=args.eps, min_samples=args.min_samples)
    save_labels(con, ids, labels)
    n_clusters, n_noise = summarize(labels)
    print(f"{args.method}: {n_clusters} clusters, {n_noise} noise faces. Saved to DB.")


def cmd_tune(con, args):
    _, X = load_embeddings(con)
    print(f"{'eps':>6} {'clusters':>9} {'noise':>6}")
    for eps in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        n_clusters, n_noise = summarize(cluster_faces(X, eps=eps, min_samples=args.min_samples))
        print(f"{eps:>6.2f} {n_clusters:>9} {n_noise:>6}")


def cmd_compare(con, args):
    _, X = load_embeddings(con)
    db = cluster_faces(X, eps=args.eps, min_samples=args.min_samples)
    hdb = cluster_faces_hdbscan(X, min_cluster_size=args.min_samples)
    for name, labels in [("DBSCAN", db), ("HDBSCAN", hdb)]:
        n_clusters, n_noise = summarize(labels)
        print(f"{name:<8} {n_clusters} clusters, {n_noise} noise faces")
    print(f"Agreement between the two (adjusted Rand, 1.0 = identical): "
          f"{adjusted_rand_score(db, hdb):.3f}")


def cmd_export(con, args):
    """Save a few face crops per cluster so you can check them by eye."""
    if DEBUG_DIR.exists():
        shutil.rmtree(DEBUG_DIR)
    rows = con.execute(
        "SELECT f.id, f.cluster_id, p.path, f.x1, f.y1, f.x2, f.y2 "
        "FROM faces f JOIN photos p ON p.id = f.photo_id "
        "WHERE f.cluster_id IS NOT NULL ORDER BY f.cluster_id"
    ).fetchall()
    saved = {}
    for face_id, cluster_id, photo_path, x1, y1, x2, y2 in rows:
        if saved.get(cluster_id, 0) >= args.per_cluster:
            continue
        folder = DEBUG_DIR / ("noise" if cluster_id == -1 else f"person_{cluster_id:03d}")
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(photo_path) as img:
                img.convert("RGB").crop((x1, y1, x2, y2)).save(folder / f"face_{face_id}.jpg")
        except OSError as err:
            print(f"Skipped {photo_path}: {err}")
            continue
        saved[cluster_id] = saved.get(cluster_id, 0) + 1
    print(f"Exported samples for {len(saved)} groups to {DEBUG_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Cluster faces into people")
    parser.add_argument("command", choices=["run", "tune", "compare", "export"])
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--method", choices=["dbscan", "hdbscan"], default="dbscan")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--per-cluster", type=int, default=8)
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    ensure_cluster_column(con)
    {"run": cmd_run, "tune": cmd_tune, "compare": cmd_compare, "export": cmd_export}[args.command](con, args)
    con.close()


if __name__ == "__main__":
    main()