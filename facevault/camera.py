"""Live webcam loop: detection, tracking and per-track recognition."""
import argparse
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from facevault.recognizer import UNKNOWN, Recognizer
from facevault.tracker import IoUTracker

GREEN = (0, 200, 0)   # OpenCV colours are BGR
RED = (0, 0, 230)


def load_analyzer(det_size=320):
    """Same model as the indexer (buffalo_l) so embeddings are comparable."""
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def detect(app, frame, min_score=0.6):
    """Return boxes (N,4) and one normalised embedding per face."""
    boxes, embeddings = [], []
    for face in app.get(frame):
        if face.det_score < min_score:
            continue
        boxes.append(face.bbox)
        embeddings.append(face.normed_embedding)
    return np.array(boxes, dtype=float).reshape(-1, 4), embeddings


def draw(frame, labels, fps, every):
    for (x1, y1, x2, y2), track_id, name, sim in labels:
        color = RED if name == UNKNOWN else GREEN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{name} #{track_id} ({sim:.2f})"
        cv2.putText(frame, text, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    lines = [f"FPS: {fps:.1f}", f"Faces: {len(labels)}", f"Detect every {every} frame(s)"]
    for i, text in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="FaceVault live camera")
    parser.add_argument("--camera", type=int, default=0, help="0 = default webcam")
    parser.add_argument("--every", type=int, default=3, help="run detection every N frames")
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--db", default="facevault.db")
    args = parser.parse_args()

    print("Loading models...")
    app = load_analyzer(args.det_size)
    recognizer = Recognizer.from_db(args.db, threshold=args.threshold)
    print(f"Known people: {', '.join(recognizer.names) or 'none (name someone in the app first)'}")
    tracker = IoUTracker(iou_threshold=0.3, max_age=30)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Check camera permissions for your terminal.")

    labels = []
    fps = 0.0
    frame_idx = 0
    last = time.perf_counter()
    print("Running. Press q to quit, r to reload names from the database.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            frame = cv2.flip(frame, 1)

            if frame_idx % args.every == 0:
                boxes, embeddings = detect(app, frame)
                tracks = tracker.update(boxes, embeddings)
                labels = []
                for t in tracks:
                    name, sim = recognizer.label(t.embeddings)
                    bbox = tuple(int(v) for v in t.bbox)
                    labels.append((bbox, t.track_id, name, sim))
            frame_idx += 1

            now = time.perf_counter()
            instant = 1.0 / max(now - last, 1e-6)
            fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
            last = now

            draw(frame, labels, fps, args.every)
            cv2.imshow("FaceVault", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                recognizer = Recognizer.from_db(args.db, threshold=args.threshold)
                print(f"Reloaded. Known people: {', '.join(recognizer.names) or 'none'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()