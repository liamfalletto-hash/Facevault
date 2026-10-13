# FaceVault

Local-first face recognition: organize your photo library by person and run a
privacy-respecting home camera that recognizes and tracks household members.
Nothing ever leaves your computer.

> 🚧 Work in progress

## Planned features
- [x] Face detection and embeddings (InsightFace)
- [x] Incremental photo indexing
- [ ] Clustering faces into people
- [ ] Streamlit app to name, merge and search people
- [ ] Live camera with tracking and recognition
- [ ] Alerts for unknown people

## Setup
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Privacy
All processing is local. No photos, faces or embeddings are uploaded anywhere.


## Face clustering

Once faces are detected and embedded, FaceVault groups them into people without
being told who anyone is. Each face is represented by a 512-dimensional
embedding, and faces whose embeddings are close together (cosine distance) are
assigned to the same cluster. Faces that fit no group are labelled as noise
(`cluster_id = -1`).

### Usage

```bash
python -m facevault.cluster tune                  # try several eps values
python -m facevault.cluster run --eps 0.5         # cluster and save to the DB
python -m facevault.cluster run --method hdbscan  # alternative algorithm
python -m facevault.cluster compare --eps 0.5     # DBSCAN vs HDBSCAN
python -m facevault.cluster export                # save sample crops per cluster
```

### Results

Tested on a subset of the [LFW dataset](http://vis-www.cs.umass.edu/lfw/):
100 people, up to 20 photos each, 1,063 detected faces.

**Tuning DBSCAN's `eps`**

| eps  | Clusters | Noise faces |
|------|----------|-------------|
| 0.35 | 99       | 231         |
| 0.40 | 100      | 173         |
| 0.45 | 101      | 147         |
| 0.50 | 101      | 136         |
| 0.55 | 101      | 135         |
| 0.60 | 102      | 130         |

`eps` is the maximum distance at which two faces count as neighbours. Across
the tested range the number of clusters stays within 99 to 102 for 100 true
identities, so the result is stable. Noise falls quickly up to 0.50 and then
levels off, so 0.50 was chosen: larger values recover almost no extra faces
while raising the risk of merging different people.

**DBSCAN vs HDBSCAN**

| Method              | Clusters | Noise faces | Noise share |
|---------------------|----------|-------------|-------------|
| DBSCAN (eps = 0.50) | 101      | 136         | 12.8%       |
| HDBSCAN             | 103      | 111         | 10.4%       |

Agreement between the two methods (adjusted Rand index): 0.875

Both methods recover close to the true number of identities and largely agree
with each other. HDBSCAN needs no `eps` tuning and assigns more faces to a
cluster, at the cost of slightly more clusters than there are people, which
suggests a few identities are split. FaceVault uses DBSCAN with `eps = 0.50` by
default, since its cluster count is closest to the true 100, and offers HDBSCAN
through `--method hdbscan`.

### Limitations

- These figures compare cluster counts, not per-face accuracy: a matching
  count does not guarantee every cluster contains a single person.
- A cluster needs at least 3 faces, so people with fewer photos are treated as
  noise. Part of the remaining noise is background faces in the photos.
- Profile shots, heavy occlusion and low-resolution faces are the most common
  source of errors.
- The best `eps` depends on the embedding model and the photo collection, so it
  should be re-tuned on new data.

### Privacy

Everything runs locally. The photo folders, the database and the exported
sample crops are excluded from version control.