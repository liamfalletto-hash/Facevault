import sqlite3

import numpy as np
import pytest

from facevault.db import FaceDB, blob_to_embedding, embedding_to_blob


def random_embedding(seed=0):
    """A fake normalized 512-number embedding."""
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def db(tmp_path):
    # a fresh, empty database in a temporary folder for each test
    with FaceDB(tmp_path / "test.db") as d:
        yield d


def test_embedding_roundtrip():
    e = random_embedding()
    assert np.array_equal(blob_to_embedding(embedding_to_blob(e)), e)


def test_wrong_embedding_size_is_rejected():
    with pytest.raises(ValueError):
        embedding_to_blob(np.zeros(10))


def test_add_photo_and_face(db):
    photo_id = db.add_photo("a.jpg", "hash_a")
    assert db.has_photo("hash_a")
    db.add_face(photo_id, (1, 2, 3, 4), 0.9, random_embedding())
    ids, matrix = db.all_embeddings()
    assert len(ids) == 1
    assert matrix.shape == (1, 512)


def test_duplicate_photo_is_rejected(db):
    db.add_photo("a.jpg", "hash_a")
    with pytest.raises(sqlite3.IntegrityError):
        db.add_photo("copy_of_a.jpg", "hash_a")   # same content, different name


def test_deleting_photo_deletes_its_faces(db):
    photo_id = db.add_photo("a.jpg", "hash_a")
    db.add_face(photo_id, (1, 2, 3, 4), 0.9, random_embedding())
    with db.conn:
        db.conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
    assert db.stats()["faces"] == 0


def test_search_photos_by_person(db):
    p1 = db.add_photo("beach.jpg", "hash_1")
    p2 = db.add_photo("office.jpg", "hash_2")
    f1 = db.add_face(p1, (0, 0, 10, 10), 0.9, random_embedding(1))
    db.add_face(p2, (0, 0, 10, 10), 0.9, random_embedding(2))

    db.set_clusters([f1], [0])               # pretend clustering put f1 in cluster 0
    maman = db.add_person("Maman")
    db.name_cluster(0, maman)

    photos = db.photos_with_person(maman)
    assert [p["path"] for p in photos] == ["beach.jpg"]