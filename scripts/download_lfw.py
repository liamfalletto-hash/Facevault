from pathlib import Path
import shutil

from sklearn.datasets import fetch_lfw_people

fetch_lfw_people()  # downloads and caches the dataset (about 200 MB)

src = Path.home() / "scikit_learn_data/lfw_home/lfw_funneled"
dst = Path("faces_db")

# 100 people with at least 5 photos each
people = sorted(
    d for d in src.iterdir()
    if d.is_dir() and len(list(d.glob("*.jpg"))) >= 5
)[:100]

for d in people:
    target = dst / d.name
    target.mkdir(parents=True, exist_ok=True)
    for photo in sorted(d.glob("*.jpg"))[:20]:
        shutil.copy2(photo, target / photo.name)

print(f"{len(people)} people copied to {dst}/")