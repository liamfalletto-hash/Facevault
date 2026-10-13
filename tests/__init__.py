import numpy as np
import pytest

from facevault.tracker import IoUTracker, iou_matrix


def box(x, y, size=100):
    return [x, y, x + size, y + size]


def test_iou_values():
    assert iou_matrix([box(0, 0)], [box(0, 0)])[0, 0] == pytest.approx(1.0)
    assert iou_matrix([box(0, 0)], [box(500, 500)])[0, 0] == 0.0
    # Half overlap: intersection 5000, union 15000
    assert iou_matrix([box(0, 0)], [box(50, 0)])[0, 0] == pytest.approx(1 / 3)


def test_moving_face_keeps_its_id():
    tracker = IoUTracker()
    ids = set()
    for frame in range(50):
        tracks = tracker.update([box(100 + 3 * frame, 100 + 2 * frame)])
        ids.add(tracks[0].track_id)
    assert ids == {1}


def test_two_faces_keep_separate_ids():
    tracker = IoUTracker()
    for frame in range(30):
        tracks = tracker.update([box(50 + 2 * frame, 100), box(400 - 2 * frame, 100)])
        assert sorted(t.track_id for t in tracks) == [1, 2]
    by_id = {t.track_id: t.bbox[0] for t in tracks}
    assert by_id[1] < by_id[2]  # face 1 is still the one on the left


def test_order_of_detections_does_not_matter():
    tracker = IoUTracker()
    tracker.update([box(0, 0), box(400, 0)])
    tracks = tracker.update([box(402, 0), box(2, 0)])  # same faces, swapped order
    by_id = {t.track_id: t.bbox[0] for t in tracks}
    assert by_id[1] == 2 and by_id[2] == 402


def test_new_face_gets_new_id():
    tracker = IoUTracker()
    tracker.update([box(0, 0)])
    tracks = tracker.update([box(0, 0), box(600, 600)])
    assert sorted(t.track_id for t in tracks) == [1, 2]


def test_short_gap_keeps_id():
    tracker = IoUTracker(max_age=30)
    tracker.update([box(100, 100)])
    for _ in range(10):
        tracker.update([])  # face missed for 10 frames
    tracks = tracker.update([box(102, 101)])
    assert tracks[0].track_id == 1


def test_long_gap_creates_new_id():
    tracker = IoUTracker(max_age=30)
    tracker.update([box(100, 100)])
    for _ in range(31):
        tracker.update([])
    assert tracker.tracks == []
    tracks = tracker.update([box(100, 100)])
    assert tracks[0].track_id == 2


def test_low_overlap_is_not_matched():
    tracker = IoUTracker(iou_threshold=0.3)
    tracker.update([box(0, 0)])
    tracks = tracker.update([box(80, 0)])  # IoU about 0.11
    assert tracks[0].track_id == 2