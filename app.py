"""FaceVault UI: name people, fix clustering mistakes, search photos."""
import sqlite3

import streamlit as st
from PIL import Image

DB_PATH = "facevault.db"
THUMB = 96          # face thumbnail size in pixels
PER_PAGE = 10       # clusters shown per page on the People page


# ---------- database helpers ----------

def get_con():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def list_clusters(con):
    """One row per cluster: id, number of faces, current name (or None)."""
    return con.execute(
        """
        SELECT f.cluster_id, COUNT(*) AS n,
               (SELECT p.name FROM faces f2
                JOIN persons p ON p.id = f2.person_id
                WHERE f2.cluster_id = f.cluster_id
                GROUP BY p.id ORDER BY COUNT(*) DESC LIMIT 1) AS name
        FROM faces f
        WHERE f.cluster_id IS NOT NULL AND f.cluster_id != -1
        GROUP BY f.cluster_id
        ORDER BY n DESC
        """
    ).fetchall()


def faces_in_cluster(con, cluster_id, limit=-1):
    return con.execute(
        """
        SELECT f.id, p.path, f.x1, f.y1, f.x2, f.y2
        FROM faces f JOIN photos p ON p.id = f.photo_id
        WHERE f.cluster_id = ?
        ORDER BY f.score DESC LIMIT ?
        """,
        (cluster_id, limit),
    ).fetchall()


def name_cluster(con, cluster_id, name):
    """Create the person if needed, then attach every face in the cluster."""
    con.execute("INSERT OR IGNORE INTO persons (name) VALUES (?)", (name,))
    person_id = con.execute("SELECT id FROM persons WHERE name = ?", (name,)).fetchone()[0]
    con.execute("UPDATE faces SET person_id = ? WHERE cluster_id = ?", (person_id, cluster_id))
    con.commit()


def merge_clusters(con, keep, drop):
    """Move every face from `drop` into `keep`. Keeps `keep`'s name if it has one."""
    row = con.execute(
        """
        SELECT person_id FROM faces
        WHERE cluster_id IN (?, ?) AND person_id IS NOT NULL
        ORDER BY (cluster_id = ?) DESC LIMIT 1
        """,
        (keep, drop, keep),
    ).fetchone()
    con.execute("UPDATE faces SET cluster_id = ? WHERE cluster_id = ?", (keep, drop))
    if row:
        con.execute("UPDATE faces SET person_id = ? WHERE cluster_id = ?", (row[0], keep))
    con.commit()


def remove_face(con, face_id):
    """Take a wrong face out of its cluster (it becomes noise, with no name)."""
    con.execute("UPDATE faces SET cluster_id = -1, person_id = NULL WHERE id = ?", (face_id,))
    con.commit()


# ---------- image helpers ----------

@st.cache_data(show_spinner=False)
def face_thumb(path, x1, y1, x2, y2):
    with Image.open(path) as img:
        crop = img.convert("RGB").crop((x1, y1, x2, y2))
    return crop.resize((THUMB, THUMB))


@st.cache_data(show_spinner=False)
def photo_preview(path):
    with Image.open(path) as img:
        preview = img.convert("RGB")
    preview.thumbnail((500, 500))
    return preview


def cluster_label(c):
    return f"{c['name'] or 'Unnamed'} · cluster {c['cluster_id']} · {c['n']} faces"


# ---------- page 1: People ----------

def page_people(con):
    st.header("People")
    clusters = list_clusters(con)
    if not clusters:
        st.info("No clusters yet. Run: python -m facevault.cluster run")
        return

    n_pages = (len(clusters) - 1) // PER_PAGE + 1
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1)
    st.caption(f"{len(clusters)} clusters, page {page} of {n_pages}")

    for c in clusters[(page - 1) * PER_PAGE : page * PER_PAGE]:
        cid = c["cluster_id"]
        st.subheader(cluster_label(c))
        cols = st.columns(8)
        for col, f in zip(cols, faces_in_cluster(con, cid, limit=8)):
            col.image(face_thumb(f["path"], f["x1"], f["y1"], f["x2"], f["y2"]))

        left, right = st.columns([4, 1])
        name = left.text_input(
            "Name", value=c["name"] or "", key=f"name_{cid}",
            label_visibility="collapsed", placeholder="Name this person",
        )
        if right.button("Save", key=f"save_{cid}"):
            if name.strip():
                name_cluster(con, cid, name.strip())
                st.rerun()
            else:
                st.warning("Type a name first.")
        st.divider()


# ---------- page 2: Fix mistakes ----------

def page_fix(con):
    st.header("Fix mistakes")
    clusters = list_clusters(con)
    if len(clusters) < 2:
        st.info("Not enough clusters to fix anything yet.")
        return
    by_id = {c["cluster_id"]: c for c in clusters}
    ids = list(by_id)
    fmt = lambda cid: cluster_label(by_id[cid])

    st.subheader("Merge two clusters")
    st.caption("Use this when the same person was split into two clusters.")
    keep = st.selectbox("Keep this cluster", ids, format_func=fmt, key="merge_keep")
    drop = st.selectbox("Merge this one into it", ids, format_func=fmt, key="merge_drop", index=1)
    for cid in (keep, drop):
        cols = st.columns(8)
        for col, f in zip(cols, faces_in_cluster(con, cid, limit=8)):
            col.image(face_thumb(f["path"], f["x1"], f["y1"], f["x2"], f["y2"]))
    if st.button("Merge"):
        if keep == drop:
            st.warning("Pick two different clusters.")
        else:
            merge_clusters(con, keep, drop)
            st.rerun()

    st.divider()
    st.subheader("Remove a wrong face")
    st.caption("Use this when a cluster contains a face of someone else.")
    target = st.selectbox("Cluster", ids, format_func=fmt, key="remove_from")
    faces = faces_in_cluster(con, target)
    cols = st.columns(8)
    for i, f in enumerate(faces):
        col = cols[i % 8]
        col.image(face_thumb(f["path"], f["x1"], f["y1"], f["x2"], f["y2"]))
        if col.button("Remove", key=f"rm_{f['id']}"):
            remove_face(con, f["id"])
            st.rerun()


# ---------- page 3: Search ----------

def page_search(con):
    st.header("Search")
    persons = con.execute("SELECT id, name FROM persons ORDER BY name").fetchall()
    if not persons:
        st.info("Name some people on the People page first.")
        return
    name_to_id = {p["name"]: p["id"] for p in persons}
    chosen = st.multiselect("Photos containing all of these people", list(name_to_id))

    use_dates = st.checkbox("Filter by date")
    date_from = date_to = None
    if use_dates:
        c1, c2 = st.columns(2)
        date_from = c1.date_input("From")
        date_to = c2.date_input("To")

    if not chosen:
        return

    person_ids = [name_to_id[n] for n in chosen]
    marks = ",".join("?" * len(person_ids))
    sql = f"""
        SELECT p.id, p.path, p.taken_at
        FROM photos p JOIN faces f ON f.photo_id = p.id
        WHERE f.person_id IN ({marks})
    """
    params = list(person_ids)
    if use_dates:
        # works for both "2024-05-01T12:00:00" and EXIF-style "2024:05:01 12:00:00"
        sql += " AND substr(replace(p.taken_at, ':', '-'), 1, 10) BETWEEN ? AND ?"
        params += [date_from.isoformat(), date_to.isoformat()]
    sql += " GROUP BY p.id HAVING COUNT(DISTINCT f.person_id) = ? ORDER BY p.taken_at"
    params.append(len(person_ids))

    photos = con.execute(sql, params).fetchall()
    st.caption(f"{len(photos)} photo(s) with " + " AND ".join(chosen))
    cols = st.columns(4)
    for i, p in enumerate(photos):
        cols[i % 4].image(photo_preview(p["path"]), caption=p["taken_at"] or "no date")


# ---------- main ----------

def main():
    st.set_page_config(page_title="FaceVault", layout="wide")
    st.sidebar.title("FaceVault")
    page = st.sidebar.radio("Page", ["People", "Fix mistakes", "Search"])
    con = get_con()
    {"People": page_people, "Fix mistakes": page_fix, "Search": page_search}[page](con)
    con.close()


if __name__ == "__main__":
    main()