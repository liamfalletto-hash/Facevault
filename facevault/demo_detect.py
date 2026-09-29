"""Draw a box around every face in an image.

Usage: python -m facevault.demo_detect photos/me1.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from facevault.detector import FaceEngine


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m facevault.demo_detect <image>")
        sys.exit(1)

    path = Path(sys.argv[1])
    img = cv2.imread(str(path))
    if img is None:
        print(f"Could not read {path}. Check the path, and convert HEIC photos to JPEG.")
        sys.exit(1)

    engine = FaceEngine()
    faces = engine.analyze(img)
    print(f"Found {len(faces)} face(s) in {path.name}")

    thickness = max(2, img.shape[1] // 400)   # thicker lines on big photos
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(img, f"{face.score:.2f}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, thickness / 2, (0, 255, 0), thickness)
        print(f"  face {i}: box={face.bbox}  score={face.score:.2f}  "
              f"embedding shape={face.embedding.shape}  "
              f"length={np.linalg.norm(face.embedding):.2f}")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{path.stem}_faces.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"Saved result to {out_path}")


if __name__ == "__main__":
    main()