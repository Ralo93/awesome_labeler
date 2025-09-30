import streamlit as st
from pathlib import Path
from io import BytesIO
import sys
import pandas as pd
import numpy as np
from pyarrow import parquet as pq
# Optional: if you have shared app modules
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from app.state import AppState  # noqa: F401
except Exception:
    AppState = None

st.set_page_config(page_title="Parquet Inspector", page_icon="🧮", layout="wide")

st.title("🧮 04 · Parquet Inspector")
st.caption("Upload or pick a Parquet file, browse rows, and inspect schema & summary statistics.")

# -----------------------------
# Helpers & Caching
# -----------------------------
@st.cache_data(show_spinner=False)
def load_parquet_bytes(data: bytes, columns=None) -> pd.DataFrame:
    bio = BytesIO(data)
    return pd.read_parquet(bio, engine="pyarrow", columns=columns)

@st.cache_data(show_spinner=False)
def get_schema_info(data: bytes):
    
    bio = BytesIO(data)
    pf = pq.ParquetFile(bio)
    schema = pf.schema_arrow
    meta = {
        "num_row_groups": pf.num_row_groups,
        "serialized_size": getattr(pf.metadata, "serialized_size", None),
        "created_by": getattr(pf.metadata, "created_by", None),
    }
    fields = []
    for f in schema:
        fields.append({
            "name": f.name,
            "type": str(f.type),
            "nullable": f.nullable,
            "metadata": {k.decode(): v.decode() for k, v in (f.metadata or {}).items()}
        })
    return schema, meta, fields

@st.cache_data(show_spinner=False)
def df_memory_bytes(df: pd.DataFrame) -> int:
    return int(df.memory_usage(deep=True).sum())

# -----------------------------
# Source selection
# -----------------------------
left, right = st.columns([1, 2])
with left:
    st.subheader("Source")
    mode = st.radio("Choose input method", ["Upload", "Pick from folder"], horizontal=True)

    default_dir = st.text_input(
        "Folder to scan for .parquet / .pq",
        value=str((Path.cwd() / "data" / "parquets").resolve()),
        help="Used when 'Pick from folder' is selected."
    )

    uploaded_bytes = None
    picked_path = None

    if mode == "Upload":
        uf = st.file_uploader("Upload a Parquet file", type=["parquet", "pq"]) 
        if uf is not None:
            uploaded_bytes = uf.read()
            st.success(f"Loaded file: {uf.name} ({len(uploaded_bytes)/1024:.1f} KB)")
    else:
        folder = Path(default_dir)
        files = []
        if folder.exists():
            files = sorted([p for p in folder.rglob("*") if p.suffix.lower() in {".parquet", ".pq"}])
        picked = st.selectbox("Pick a file", options=files, format_func=lambda p: str(p.relative_to(folder)) if files else str)
        if picked:
            picked_path = Path(picked)
            try:
                uploaded_bytes = picked_path.read_bytes()
            except Exception as e:
                st.error(f"Failed to read file: {e}")

with right:
    st.subheader("Options")
    sample_rows = st.number_input("Preview rows (per page)", min_value=10, max_value=50_000, value=1_000, step=100)
    page_index = st.number_input("Page index (0-based)", min_value=0, value=0, step=1)
    show_index = st.toggle("Show DataFrame index", value=False)
    enable_filter = st.toggle("Enable quick text filter", value=False, help="Case-insensitive contains across selected columns")

st.divider()

if uploaded_bytes is None:
    st.info("Upload or pick a Parquet to begin.")
    st.stop()

# -----------------------------
# Schema & basic metadata
# -----------------------------
schema, meta, fields = get_schema_info(uploaded_bytes)

meta_cols = st.columns(4)
with meta_cols[0]:
    st.metric("Row Groups", meta.get("num_row_groups", "-"))
with meta_cols[1]:
    st.metric("Fields", len(fields))
with meta_cols[2]:
    st.metric("Creator", meta.get("created_by") or "—")
with meta_cols[3]:
    st.metric("File Size (KB)", f"{len(uploaded_bytes)/1024:.1f}")

with st.expander("📜 Schema fields"):
    st.dataframe(pd.DataFrame(fields), use_container_width=True, hide_index=True)

# -----------------------------
# Column selection + lazy load
# -----------------------------
all_cols = [f["name"] for f in fields]
with st.container(border=True):
    st.subheader("Columns to load")
    sel_cols = st.multiselect("Select columns (empty = all)", options=all_cols, default=[])
    if enable_filter:
        filter_cols = st.multiselect("Columns to apply text filter on", options=sel_cols or all_cols, default=sel_cols or all_cols)
        query_text = st.text_input("Filter value (contains, case-insensitive)", value="")
    else:
        filter_cols, query_text = [], ""

# Load dataframe
with st.spinner("Reading parquet …"):
    df = load_parquet_bytes(uploaded_bytes, columns=(sel_cols or None))

# -----------------------------
# High-level stats
# -----------------------------
mem_mb = df_memory_bytes(df) / (1024**2)
stat_cols = st.columns(6)
with stat_cols[0]:
    st.metric("Rows", f"{len(df):,}")
with stat_cols[1]:
    st.metric("Columns", df.shape[1])
with stat_cols[2]:
    st.metric("Memory (MB)", f"{mem_mb:.2f}")
with stat_cols[3]:
    st.metric("Numeric cols", df.select_dtypes(include=np.number).shape[1])
with stat_cols[4]:
    st.metric("String cols", df.select_dtypes(include="object").shape[1])
with stat_cols[5]:
    dupes = int(df.duplicated().sum())
    st.metric("Duplicate rows", f"{dupes:,}")

# -----------------------------
# Quick profiling tabs
# -----------------------------
prof_tabs = st.tabs(["Data", "Summary", "Missingness", "Types", "Correlations"]) 

# Data tab with paging & filtering
with prof_tabs[0]:
    _df = df
    if enable_filter and query_text.strip():
        qt = query_text.strip().lower()
        mask = np.zeros(len(_df), dtype=bool)
        for c in filter_cols:
            try:
                s = _df[c].astype(str).str.lower().str.contains(qt, na=False)
                mask = mask | s.values
            except Exception:
                pass
        _df = _df[mask]
        st.caption(f"Filtered rows: {_df.shape[0]:,} (of {df.shape[0]:,})")

    start = int(page_index * sample_rows)
    end = int(min(start + sample_rows, len(_df)))
    if start >= len(_df):
        st.warning("Page index beyond data length. Showing last page.")
        last_page = max((len(_df) - 1) // sample_rows, 0)
        start = last_page * sample_rows
        end = len(_df)
    view = _df.iloc[start:end]

    st.dataframe(view, use_container_width=True, hide_index=not show_index)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⬅️ Prev page"):
            st.session_state["__pgidx"] = max(int(page_index) - 1, 0)
            st.rerun()
    with c2:
        st.write(f"Page: {int(page_index)} | Rows {start:,}–{end:,}")
    with c3:
        if st.button("Next page ➡️"):
            st.session_state["__pgidx"] = int(page_index) + 1
            st.rerun()

    st.download_button("⬇️ Download current view (CSV)", data=view.to_csv(index=show_index).encode("utf-8"), file_name="view.csv", mime="text/csv")

# Summary stats tab
with prof_tabs[1]:
    st.subheader("Numeric summary")
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] > 0:
        st.dataframe(num_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T, use_container_width=True)
    else:
        st.info("No numeric columns.")

    st.subheader("Categorical (object) top values")
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    top_k = st.slider("Show top K per column", 3, 30, 10, step=1)
    if obj_cols:
        for c in obj_cols:
            with st.expander(f"{c}"):
                vc = df[c].value_counts(dropna=False).head(top_k)
                tbl = pd.DataFrame({"value": vc.index.astype(str), "count": vc.values})
                st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        st.info("No object (string) columns.")

# Missingness tab
with prof_tabs[2]:
    miss = df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / len(df) * 100).round(2)
    miss_tbl = pd.DataFrame({"missing": miss, "missing_%": miss_pct})
    st.dataframe(miss_tbl, use_container_width=True)
    try:
        st.bar_chart(miss_pct.rename("missing_%"))
    except Exception:
        pass

# Types tab
with prof_tabs[3]:
    dtypes_tbl = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
    st.dataframe(dtypes_tbl, use_container_width=True, hide_index=True)

    by_type = dtypes_tbl.groupby("dtype").size().sort_values(ascending=False)
    st.bar_chart(by_type)

# Correlations tab
with prof_tabs[4]:
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)
        st.dataframe(corr, use_container_width=True)
    else:
        st.info("Need at least two numeric columns for correlations.")

st.divider()

# Lightweight column inspector
st.subheader("🔎 Inspect a column")
col_to_inspect = st.selectbox("Column", options=df.columns.tolist())
if col_to_inspect:
    c = df[col_to_inspect]
    left, right = st.columns([2, 1])
    with left:
        st.write(pd.DataFrame({
            "dtype": [str(c.dtype)],
            "unique": [int(c.nunique(dropna=True))],
            "missing": [int(c.isna().sum())],
            "min": [c.min() if np.issubdtype(c.dtype, np.number) else None],
            "max": [c.max() if np.issubdtype(c.dtype, np.number) else None],
            "mean": [c.mean() if np.issubdtype(c.dtype, np.number) else None],
        }))
    with right:
        try:
            if np.issubdtype(c.dtype, np.number):
                st.histogram(c.dropna(), bins=30)
            else:
                st.bar_chart(c.astype(str).value_counts().head(20))
        except Exception:
            pass

# Footer
st.caption("Tip: For very large files, select only the columns you need, and use the page size to limit the view.")
