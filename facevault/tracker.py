"""Phase 8: simple IoU tracker with Hungarian matching."""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    """IoU between every box in a (N,4) and every box in b (M,4). Boxes are x1,y1,x2,y2."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    # Overlap rectangle, computed for all pairs at once via broadcasting
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    embeddings: deque = field(default_factory=lambda: deque(maxlen=10))
    frames_since_seen: int = 0
    hits: int = 1  # how many times this track has been matched


class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_age=30, max_embeddings=10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.max_embeddings = max_embeddings
        self.tracks = []
        self._next_id = 1

    def _new_track(self, bbox, embedding=None):
        track = Track(self._next_id, np.asarray(bbox, dtype=float),
                      deque(maxlen=self.max_embeddings))
        if embedding is not None:
            track.embeddings.append(embedding)
        self._next_id += 1
        self.tracks.append(track)

    def update(self, boxes, embeddings=None):
        """Feed one frame of detections. Returns the tracks seen in this frame."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        if embeddings is None:
            embeddings = [None] * len(boxes)

        matched_tracks, matched_dets = set(), set()

        if self.tracks and len(boxes):
            track_boxes = np.vstack([t.bbox for t in self.tracks])
            ious = iou_matrix(track_boxes, boxes)
            rows, cols = linear_sum_assignment(1 - ious)  # Hungarian algorithm
            for r, c in zip(rows, cols):
                if ious[r, c] > self.iou_threshold:
                    track = self.tracks[r]
                    track.bbox = boxes[c]
                    track.frames_since_seen = 0
                    track.hits += 1
                    if embeddings[c] is not None:
                        track.embeddings.append(embeddings[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

        # Tracks with no match this frame get older
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.frames_since_seen += 1

        # Detections with no match start new tracks
        for c in range(len(boxes)):
            if c not in matched_dets:
                self._new_track(boxes[c], embeddings[c])

        # Forget tracks unseen for too long
        self.tracks = [t for t in self.tracks if t.frames_since_seen <= self.max_age]

        return [t for t in self.tracks if t.frames_since_seen == 0]