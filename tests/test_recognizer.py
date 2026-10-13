import numpy as np

from facevault.recognizer import UNKNOWN, Recognizer


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


ALICE, BOB = unit([1, 0, 0]), unit([0, 1, 0])
rec = Recognizer(["Alice", "Bob"], np.vstack([ALICE, BOB]), threshold=0.45, min_frames=3)


def test_consistent_frames_are_recognised():
    name, sim = rec.label([ALICE] * 5)
    assert name == "Alice" and sim > 0.99


def test_far_from_everyone_is_unknown():
    assert rec.label([unit([0, 0, 1])] * 5)[0] == UNKNOWN


def test_majority_vote_wins():
    assert rec.label([ALICE] * 7 + [BOB] * 3)[0] == "Alice"


def test_no_majority_is_unknown():
    assert rec.label([ALICE] * 5 + [BOB] * 5)[0] == UNKNOWN


def test_too_few_frames_is_unknown():
    assert rec.label([ALICE] * 2)[0] == UNKNOWN


def test_no_known_people():
    assert Recognizer([], np.empty((0, 3))).label([ALICE] * 5)[0] == UNKNOWN