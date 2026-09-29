"""Face detection and embeddings using InsightFace."""
from dataclasses import dataclass

import numpy as np
from insightface.app import FaceAnalysis


@dataclass
class DetectedFace:
    """One face found in an image."""
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    score: float                      # detection confidence, 0 to 1
    embedding: np.ndarray             # 512 numbers describing the face


class FaceEngine:
    """Finds faces in an image and turns each one into an embedding."""

    def __init__(self, det_size=(640, 640), min_score=0.6):
        self.min_score = min_score
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=det_size)

    def analyze(self, img_bgr: np.ndarray) -> list[DetectedFace]:
        """Return every face in the image with a confidence above min_score."""
        results = []
        for f in self.app.get(img_bgr):
            if f.det_score < self.min_score:
                continue
            x1, y1, x2, y2 = (max(0, int(v)) for v in f.bbox)
            results.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                score=float(f.det_score),
                embedding=f.normed_embedding,
            ))
        return results