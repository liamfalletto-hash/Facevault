"""Compare the first face in two images.

Usage: python -m facevault.demo_compare photos/me1.jpg photos/me2.jpg
"""
import sys

import cv2

from facevault.detector import FaceEngine


def first_face_embedding(engine, path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read {path}")
    faces = engine.analyze(img)
    if not faces:
        sys.exit(f"No face found in {path}")
    # if several faces, take the biggest one
    biggest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return biggest.embedding


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m facevault.demo_compare <image1> <image2>")
    engine = FaceEngine()
    a = first_face_embedding(engine, sys.argv[1])
    b = first_face_embedding(engine, sys.argv[2])
    similarity = float(a @ b)
    print(f"Cosine similarity: {similarity:.3f}")
    print("Probably the same person" if similarity > 0.4 else "Probably different people")


if __name__ == "__main__":
    main()