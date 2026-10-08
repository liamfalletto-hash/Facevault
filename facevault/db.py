"""SQLite storage for photos, faces, people and camera events."""
import sqlite3
from pathlib import Path

import numpy as np

EMBEDDING_DIM = 512

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id        INTEGER PRIMARY KEY,
    path      TEXT NOT NULL UNIQUE,
    sha1      TEXT NOT NULL UNIQUE,
    taken_at  TEXT,
    width     INTEGER,
    height    INTEGER,
    added_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS persons (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS faces (
    id          INTEGER PRIMARY KEY,
    photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    x1 INTEGER NOT NULL, y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL, y2 INTEGER NOT NULL,
    score       REAL NOT NULL,
    embedding   BLOB NOT NULL,
    cluster_id  INTEGER,
    person_id   INTEGER REFERENCES persons(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY,
    ts             TEXT NOT NULL DEFAULT (datetime('now')),
    track_id       INTEGER,
    person_id      INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    snapshot_path  TEXT
);

CREATE INDEX IF NOT EXISTS idx_faces_photo   ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person  ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);
"""


def embedding_to_blob(emb) -> bytes:
    """Convert a 512-number embedding into bytes for storage."""
    emb = np.asarray(emb, dtype=np.float32)
    if emb.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Expected shape ({EMBEDDING_DIM},), got {emb.shape}")
    return emb.tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Convert stored bytes back into a 512-number embedding."""
    return np.frombuffer(blob, dtype=np.float32).copy()


class FaceDB:
    """All reads and writes to the FaceVault database go through this class."""

    def __init__(self, path="facevault.db"):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------- photos ----------
    def add_photo(self, path, sha1, taken_at=None, width=None, height=None) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO photos (path, sha1, taken_at, width, height) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(path), sha1, taken_at, width, height),
            )
        return cur.lastrowid

    def has_photo(self, sha1) -> bool:
        row = self.conn.execute("SELECT 1 FROM photos WHERE sha1 = ?", (sha1,)).fetchone()
        return row is not None

    def get_photo(self, photo_id):
        return self.conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()

    # ---------- faces ----------
    def add_face(self, photo_id, bbox, score, embedding) -> int:
        x1, y1, x2, y2 = bbox
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO faces (photo_id, x1, y1, x2, y2, score, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (photo_id, x1, y1, x2, y2, float(score), embedding_to_blob(embedding)),
            )
        return cur.lastrowid

    def add_photo_with_faces(self, path, sha1, faces, taken_at=None,
                             width=None, height=None) -> int:
        """Save a photo and all its faces in one transaction (all or nothing)."""
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO photos (path, sha1, taken_at, width, height) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(path), sha1, taken_at, width, height),
            )
            photo_id = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO faces (photo_id, x1, y1, x2, y2, score, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(photo_id, *f.bbox, float(f.score), embedding_to_blob(f.embedding))
                 for f in faces],
            )
        return photo_id

    def all_embeddings(self):
        """Return (face_ids, matrix) where each row of the matrix is one embedding."""
        rows = self.conn.execute("SELECT id, embedding FROM faces ORDER BY id").fetchall()
        if not rows:
            return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        ids = [r["id"] for r in rows]
        matrix = np.vstack([blob_to_embedding(r["embedding"]) for r in rows])
        return ids, matrix

    def set_clusters(self, face_ids, labels):
        """Save the cluster number found for each face (Phase 4)."""
        with self.conn:
            self.conn.executemany(
                "UPDATE faces SET cluster_id = ? WHERE id = ?",
                [(int(label), int(fid)) for fid, label in zip(face_ids, labels)],
            )

    # ---------- persons ----------
    def add_person(self, name) -> int:
        with self.conn:
            cur = self.conn.execute("INSERT INTO persons (name) VALUES (?)", (name,))
        return cur.lastrowid

    def name_cluster(self, cluster_id, person_id):
        """Attach every face in a cluster to a named person (Phase 5)."""
        with self.conn:
            self.conn.execute(
                "UPDATE faces SET person_id = ? WHERE cluster_id = ?",
                (person_id, cluster_id),
            )

    def photos_with_person(self, person_id):
        return self.conn.execute(
            "SELECT DISTINCT p.* FROM photos p "
            "JOIN faces f ON f.photo_id = p.id "
            "WHERE f.person_id = ? ORDER BY p.taken_at",
            (person_id,),
        ).fetchall()

    # ---------- events ----------
    def add_event(self, track_id, person_id=None, snapshot_path=None) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO events (track_id, person_id, snapshot_path) VALUES (?, ?, ?)",
                (track_id, person_id, snapshot_path),
            )
        return cur.lastrowid

    # ---------- misc ----------
    def stats(self) -> dict:
        tables = ("photos", "faces", "persons", "events")
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}