"""Scan a folder of photos, detect faces and store them in the database.

Usage: python -m facevault.indexer ~/Pictures/facevault_test
"""
import argparse
import hashlib
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from facevault.db import FaceDB
from facevault.detector import FaceEngine

# Optional: open iPhone HEIC photos if pillow-heif is installed
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
if HEIC_SUPPORTED:
    IMAGE_EXTENSIONS |= {".heic", ".heif"}

EXIF_IFD = 0x8769            # the EXIF sub-section holding camera details
DATETIME_ORIGINAL = 36867    # when the photo was taken
DATETIME = 306               # when the file was last modified (fallback)


def find_images(folder: Path) -> list[Path]:
    """All image files in folder and its subfolders, skipping hidden files."""
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.name.startswith(".")
    )


def sha1_of(path: Path) -> str:
    """Fingerprint a file's contents, reading it in 1 MB chunks."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_taken_at(img: Image.Image, path: Path) -> str:
    """Date the photo was taken, as 'YYYY-MM-DD HH:MM:SS'."""
    exif = img.getexif()
    raw = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(DATETIME)
    if raw:
        try:
            taken = datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
            return taken.isoformat(sep=" ", timespec="seconds")
        except ValueError:
            pass   # malformed date: fall back to the file's date
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    return modified.isoformat(sep=" ", timespec="seconds")


def load_image(path: Path):
    """Open an image the right way up. Returns (bgr_array, taken_at)."""
    with Image.open(path) as im:
        taken_at = read_taken_at(im, path)
        im = ImageOps.exif_transpose(im).convert("RGB")
    bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return bgr, taken_at


def resize_for_detection(img: np.ndarray, max_side: int):
    """Shrink so the longest side is at most max_side. Returns (image, scale)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img, scale


def index_folder(folder, db_path="facevault.db", max_side=1600, min_score=0.6) -> dict:
    folder = Path(folder).expanduser().resolve()
    images = find_images(folder)
    stats = {"found": len(images), "new": 0, "faces": 0, "skipped": 0, "errors": 0}
    engine = None   # loaded only if there is something new to process
    start = time.perf_counter()

    with FaceDB(db_path) as db:
        for path in tqdm(images, desc="Indexing", unit="photo"):
            try:
                digest = sha1_of(path)
                if db.has_photo(digest):
                    stats["skipped"] += 1
                    continue

                if engine is None:
                    engine = FaceEngine(min_score=min_score)

                img, taken_at = load_image(path)
                height, width = img.shape[:2]
                small, scale = resize_for_detection(img, max_side)

                # detect on the small image, then convert boxes back to full size
                faces = [
                    replace(f, bbox=tuple(round(v / scale) for v in f.bbox))
                    for f in engine.analyze(small)
                ]

                db.add_photo_with_faces(path, digest, faces, taken_at=taken_at,
                                        width=width, height=height)
                stats["new"] += 1
                stats["faces"] += len(faces)
            except Exception as e:
                stats["errors"] += 1
                tqdm.write(f"⚠️  {path.name}: {e}")

    stats["seconds"] = time.perf_counter() - start
    return stats


def main():
    parser = argparse.ArgumentParser(description="Index a folder of photos into FaceVault.")
    parser.add_argument("folder", help="folder to scan (subfolders included)")
    parser.add_argument("--db", default="facevault.db", help="database file")
    parser.add_argument("--max-side", type=int, default=1600,
                        help="resize photos to this size for detection")
    parser.add_argument("--min-score", type=float, default=0.6,
                        help="minimum face detection confidence")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        parser.error(f"{folder} is not a folder")

    s = index_folder(folder, args.db, args.max_side, args.min_score)
    rate = s["new"] / s["seconds"] if s["seconds"] > 0 else 0
    print(f"\nDone in {s['seconds']:.1f}s")
    print(f"  {s['found']} image(s) found")
    print(f"  {s['new']} new, {s['skipped']} already indexed, {s['errors']} error(s)")
    print(f"  {s['faces']} face(s) saved  ({rate:.1f} new photos/s)")
    if not HEIC_SUPPORTED:
        print("  Tip: pip install pillow-heif to include iPhone HEIC photos")


if __name__ == "__main__":
    main()