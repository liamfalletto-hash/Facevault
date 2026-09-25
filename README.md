# FaceVault

Local-first face recognition: organize your photo library by person and run a
privacy-respecting home camera that recognizes and tracks household members.
Nothing ever leaves your computer.

> 🚧 Work in progress

## Planned features
- [ ] Face detection and embeddings (InsightFace)
- [ ] Incremental photo indexing
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