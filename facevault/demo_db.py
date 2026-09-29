"""Detect faces in an image and save them to the database.

Usage: python -m facevault.demo_db photos/me1.jpg
"""
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

from facevault.db import FaceDB
from facevault.detector import FaceEngine


def sha1_of(path: Path) -> str:
    """Fingerprint a file's contents, reading it in 1 MB chunks."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m facevault.demo_db <image>")

    path = Path(sys.argv[1])
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"Could not read {path}")

    with FaceDB() as db:
        digest = sha1_of(path)
        if db.has_photo(digest):
            print(f"{path.name} is already in the database, skipping.")
        else:
            faces = FaceEngine().analyze(img)
            height, width = img.shape[:2]
            photo_id = db.add_photo(path, digest, width=width, height=height)
            for face in faces:
                db.add_face(photo_id, face.bbox, face.score, face.embedding)
            print(f"Saved {path.name} (photo #{photo_id}) with {len(faces)} face(s)")

            # check the last embedding survived the round trip unchanged
            if faces:
                _, matrix = db.all_embeddings()
                same = np.array_equal(matrix[-1], faces[-1].embedding)
                print(f"Embedding read back identical: {same}")

        print("Database contents:", db.stats())


if __name__ == "__main__":
    main()