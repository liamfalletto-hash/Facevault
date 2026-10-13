"""Phase 7: live webcam loop with face detection and an FPS counter."""
import argparse
import time

import cv2
from insightface.app import FaceAnalysis


def load_detector(det_size=320):
    """Detection only for now: much faster than also computing embeddings."""
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def detect(app, frame):
    """Return a list of (x1, y1, x2, y2, score) for each face in the frame."""
    out = []
    for face in app.get(frame):
        x1, y1, x2, y2 = face.bbox.astype(int)
        out.append((x1, y1, x2, y2, float(face.det_score)))
    return out


def draw(frame, boxes, fps, every):
    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    lines = [f"FPS: {fps:.1f}", f"Faces: {len(boxes)}", f"Detect every {every} frame(s)"]
    for i, text in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="FaceVault live camera")
    parser.add_argument("--camera", type=int, default=0, help="0 = default webcam")
    parser.add_argument("--every", type=int, default=3, help="run detection every N frames")
    parser.add_argument("--det-size", type=int, default=320)
    args = parser.parse_args()

    print("Loading detector...")
    app = load_detector(args.det_size)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Check camera permissions for your terminal.")

    boxes = []
    fps = 0.0
    frame_idx = 0
    last = time.perf_counter()
    print("Running. Press q in the video window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            frame = cv2.flip(frame, 1)  # mirror image, feels natural on a webcam

            # Detect every N frames; in between, keep showing the last boxes
            if frame_idx % args.every == 0:
                boxes = detect(app, frame)
            frame_idx += 1

            now = time.perf_counter()
            instant = 1.0 / max(now - last, 1e-6)
            fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant  # smoothed
            last = now

            draw(frame, boxes, fps, args.every)
            cv2.imshow("FaceVault", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()