import numpy as np
import pytest

from facevault.detector import FaceEngine


@pytest.fixture(scope="module")
def engine():
    # load the models once for all tests in this file
    return FaceEngine()


def test_blank_image_has_no_faces(engine):
    img = np.zeros((480, 640, 3), dtype=np.uint8)   # a black 640x480 image
    assert engine.analyze(img) == []


def test_min_score_is_respected():
    engine = FaceEngine(min_score=0.99)
    assert engine.min_score == 0.99