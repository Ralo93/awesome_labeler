import streamlit as st
from pathlib import Path
from io import BytesIO
import sys
import pandas as pd
import numpy as np
from pyarrow import parquet as pq
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Optional: if you have shared app modules
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from app.state import AppState  # noqa: F401
except Exception:
    AppState = None

st.set_page_config(page_title="Parquet Inspector & Cleaner", page_icon="🧮", layout="wide")

st.title("🧮 04 · Parquet Inspector & Data Cleaner")
st.caption("Upload or pick a Parquet file, browse rows, detect outliers, and clean data for model training.")

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
# Outlier Detection Functions
# -----------------------------
def detect_outliers_iqr(df: pd.DataFrame, columns: list, multiplier: float = 1.5):
    """Detect outliers using Interquartile Range method"""
    outliers = pd.DataFrame(index=df.index)
    outliers['is_outlier'] = False
    outlier_details = {}
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            
            col_outliers = (df[col] < lower) | (df[col] > upper)
            outliers[f'{col}_outlier'] = col_outliers
            outliers['is_outlier'] = outliers['is_outlier'] | col_outliers
            
            outlier_details[col] = {
                'lower_bound': lower,
                'upper_bound': upper,
                'outlier_count': col_outliers.sum()
            }
    
    return outliers, outlier_details

def detect_outliers_zscore(df: pd.DataFrame, columns: list, threshold: float = 3):
    """Detect outliers using Z-score method"""
    outliers = pd.DataFrame(index=df.index)
    outliers['is_outlier'] = False
    outlier_details = {}
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            col_outliers = pd.Series(False, index=df.index)
            col_outliers[df[col].notna()] = z_scores > threshold
            
            outliers[f'{col}_outlier'] = col_outliers
            outliers['is_outlier'] = outliers['is_outlier'] | col_outliers
            
            outlier_details[col] = {
                'threshold': threshold,
                'outlier_count': col_outliers.sum()
            }
    
    return outliers, outlier_details

def detect_outliers_isolation(df: pd.DataFrame, columns: list, contamination: float = 0.1):
    """Detect outliers using Isolation Forest"""
    numeric_df = df[columns].select_dtypes(include=np.number)
    
    if numeric_df.shape[1] == 0:
        return pd.DataFrame({'is_outlier': [False] * len(df)}, index=df.index), {}
    
    # Handle missing values
    numeric_df_filled = numeric_df.fillna(numeric_df.median())
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(numeric_df_filled)
    
    outlier_df = pd.DataFrame(index=df.index)
    outlier_df['is_outlier'] = outliers == -1
    
    outlier_details = {
        'contamination': contamination,
        'outlier_count': (outliers == -1).sum(),
        'columns_used': list(numeric_df.columns)
    }
    
    return outlier_df, outlier_details

# -----------------------------
# Data Cleaning Functions
# -----------------------------
def clean_missing_values(df: pd.DataFrame, strategy: str, columns: list = None):
    """Handle missing values with various strategies"""
    df_clean = df.copy()
    columns = columns or df.columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == "drop":
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == "forward_fill":
            df_clean[col] = df_clean[col].fillna(method='ffill')
        elif strategy == "backward_fill":
            df_clean[col] = df_clean[col].fillna(method='bfill')
        elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df_clean[col] = df_clean[col].fillna(df[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df_clean[col] = df_clean[col].fillna(df[col].median())
        elif strategy == "mode":
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
        elif strategy == "zero" and pd.api.types.is_numeric_dtype(df[col]):
            df_clean[col] = df_clean[col].fillna(0)
    
    return df_clean

def scale_features(df: pd.DataFrame, columns: list, method: str):
    """Scale numeric features"""
    df_scaled = df.copy()
    
    numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        return df_scaled
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        return df_scaled
    
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0))
    
    return df_scaled

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

# Initialize cleaned dataframe in session state
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = df.copy()

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
# Main tabs including new cleaning tab
# -----------------------------
prof_tabs = st.tabs(["Data", "Summary", "Missingness", "Types", "Correlations", "🧹 Data Cleaning", "🎯 Outlier Detection"]) 

# Data tab with paging & filtering
with prof_tabs[0]:
    _df = st.session_state.cleaned_df if st.checkbox("Show cleaned data", value=False) else df
    
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

# Data Cleaning tab
with prof_tabs[5]:
    st.header("🧹 Data Cleaning Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Handle Duplicates")
        if st.button("Remove Duplicate Rows"):
            before_rows = len(st.session_state.cleaned_df)
            st.session_state.cleaned_df = st.session_state.cleaned_df.drop_duplicates()
            after_rows = len(st.session_state.cleaned_df)
            st.success(f"Removed {before_rows - after_rows} duplicate rows")
        
        st.subheader("2. Handle Missing Values")
        missing_strategy = st.selectbox(
            "Strategy for missing values",
            ["drop", "mean", "median", "mode", "forward_fill", "backward_fill", "zero"]
        )
        missing_cols = st.multiselect(
            "Apply to columns (empty = all)",
            options=df.columns.tolist(),
            default=[]
        )
        
        if st.button("Apply Missing Value Strategy"):
            st.session_state.cleaned_df = clean_missing_values(
                st.session_state.cleaned_df,
                missing_strategy,
                missing_cols or None
            )
            st.success(f"Applied {missing_strategy} strategy to handle missing values")
    
    with col2:
        st.subheader("3. Feature Scaling")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        scale_method = st.selectbox("Scaling method", ["none", "standard", "minmax"])
        scale_cols = st.multiselect(
            "Columns to scale",
            options=numeric_cols,
            default=[]
        )
        
        if st.button("Apply Scaling") and scale_method != "none":
            st.session_state.cleaned_df = scale_features(
                st.session_state.cleaned_df,
                scale_cols,
                scale_method
            )
            st.success(f"Applied {scale_method} scaling to selected columns")
        
        st.subheader("4. Encode Categorical Variables")
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        encode_cols = st.multiselect(
            "Columns to label encode",
            options=cat_cols,
            default=[]
        )
        
        if st.button("Apply Label Encoding"):
            for col in encode_cols:
                if col in st.session_state.cleaned_df.columns:
                    le = LabelEncoder()
                    st.session_state.cleaned_df[col] = le.fit_transform(
                        st.session_state.cleaned_df[col].fillna("missing")
                    )
            st.success(f"Applied label encoding to {len(encode_cols)} columns")
    
    st.divider()
    
    # Comparison metrics
    st.subheader("📊 Cleaning Summary")
    comp_cols = st.columns(4)
    with comp_cols[0]:
        st.metric(
            "Original Rows",
            f"{len(df):,}",
            f"{len(st.session_state.cleaned_df) - len(df):,}",
            delta_color="inverse"
        )
    with comp_cols[1]:
        orig_missing = df.isna().sum().sum()
        clean_missing = st.session_state.cleaned_df.isna().sum().sum()
        st.metric(
            "Missing Values",
            f"{clean_missing:,}",
            f"{clean_missing - orig_missing:,}",
            delta_color="inverse"
        )
    with comp_cols[2]:
        orig_dupes = df.duplicated().sum()
        clean_dupes = st.session_state.cleaned_df.duplicated().sum()
        st.metric(
            "Duplicates",
            f"{clean_dupes:,}",
            f"{clean_dupes - orig_dupes:,}",
            delta_color="inverse"
        )
    with comp_cols[3]:
        st.metric(
            "Columns",
            f"{st.session_state.cleaned_df.shape[1]}",
            f"{st.session_state.cleaned_df.shape[1] - df.shape[1]:,}"
        )
    
    # Export options
    st.divider()
    st.subheader("💾 Export Cleaned Data")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_data = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
    
    with col2:
        parquet_buffer = BytesIO()
        st.session_state.cleaned_df.to_parquet(parquet_buffer, index=False)
        st.download_button(
            label="📦 Download as Parquet",
            data=parquet_buffer.getvalue(),
            file_name="cleaned_data.parquet",
            mime="application/octet-stream"
        )
    
    with col3:
        if st.button("🔄 Reset to Original"):
            st.session_state.cleaned_df = df.copy()
            st.success("Reset to original data")
            st.rerun()

# Outlier Detection tab
with prof_tabs[6]:
    st.header("🎯 Outlier Detection & Removal")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns available for outlier detection")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Detection Settings")
            
            method = st.selectbox(
                "Detection Method",
                ["IQR (Interquartile Range)", "Z-Score", "Isolation Forest"],
                help="Choose method for detecting outliers"
            )
            
            selected_cols = st.multiselect(
                "Columns to analyze",
                options=numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if method == "IQR (Interquartile Range)":
                iqr_multiplier = st.slider(
                    "IQR Multiplier",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Standard is 1.5. Lower values = more outliers detected"
                )
            elif method == "Z-Score":
                z_threshold = st.slider(
                    "Z-Score Threshold",
                    min_value=2.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.5,
                    help="Standard is 3. Lower values = more outliers detected"
                )
            else:  # Isolation Forest
                contamination = st.slider(
                    "Contamination",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.1,
                    step=0.01,
                    help="Expected proportion of outliers in the dataset"
                )
            
            detect_button = st.button("🔍 Detect Outliers", type="primary")
        
        with col2:
            if detect_button and selected_cols:
                with st.spinner("Detecting outliers..."):
                    # Use cleaned_df for detection
                    working_df = st.session_state.cleaned_df
                    
                    if method == "IQR (Interquartile Range)":
                        outliers, details = detect_outliers_iqr(working_df, selected_cols, iqr_multiplier)
                    elif method == "Z-Score":
                        outliers, details = detect_outliers_zscore(working_df, selected_cols, z_threshold)
                    else:
                        outliers, details = detect_outliers_isolation(working_df, selected_cols, contamination)
                    
                    # Store outliers in session state
                    st.session_state.outliers = outliers
                    st.session_state.outlier_details = details
            
            if 'outliers' in st.session_state:
                st.subheader("Detection Results")
                
                total_outliers = st.session_state.outliers['is_outlier'].sum()
                total_rows = len(st.session_state.outliers)
                outlier_percentage = (total_outliers / total_rows) * 100
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Total Outliers", f"{total_outliers:,}")
                with metric_cols[1]:
                    st.metric("Percentage", f"{outlier_percentage:.2f}%")
                with metric_cols[2]:
                    st.metric("Clean Rows", f"{total_rows - total_outliers:,}")
                
                # Show details per column
                if method != "Isolation Forest":
                    st.subheader("Per-Column Analysis")
                    for col, details in st.session_state.outlier_details.items():
                        with st.expander(f"📊 {col}"):
                            if method == "IQR (Interquartile Range)":
                                st.write(f"- Lower bound: {details['lower_bound']:.4f}")
                                st.write(f"- Upper bound: {details['upper_bound']:.4f}")
                            elif method == "Z-Score":
                                st.write(f"- Threshold: ±{details['threshold']}")
                            st.write(f"- Outliers found: {details['outlier_count']}")
                
                # Visualization
                st.subheader("Outlier Visualization")
                viz_col = st.selectbox("Select column to visualize", selected_cols)
                
                if viz_col and viz_col in st.session_state.cleaned_df.columns:
                    import matplotlib.pyplot as plt
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Box plot
                    data_to_plot = st.session_state.cleaned_df[viz_col].dropna()
                    ax1.boxplot(data_to_plot)
                    ax1.set_title(f'Box Plot - {viz_col}')
                    ax1.set_ylabel('Value')
                    
                    # Histogram with outliers marked
                    ax2.hist(data_to_plot, bins=50, alpha=0.7, color='blue', edgecolor='black')
                    
                    if f'{viz_col}_outlier' in st.session_state.outliers.columns:
                        outlier_mask = st.session_state.outliers[f'{viz_col}_outlier']
                        outlier_values = st.session_state.cleaned_df.loc[outlier_mask, viz_col]
                        ax2.hist(outlier_values, bins=50, alpha=0.7, color='red', edgecolor='black', label='Outliers')
                        ax2.legend()
                    
                    ax2.set_title(f'Distribution - {viz_col}')
                    ax2.set_xlabel('Value')
                    ax2.set_ylabel('Frequency')
                    
                    st.pyplot(fig)
                
                # Remove outliers option
                st.divider()
                st.subheader("🗑️ Remove Outliers")
                
                removal_option = st.radio(
                    "Removal Strategy",
                    ["Keep outliers (no action)",
                     "Remove all detected outliers",
                     "Remove outliers from specific columns"]
                )
                
                if removal_option == "Remove outliers from specific columns":
                    outlier_cols = [col for col in st.session_state.outliers.columns if col.endswith('_outlier')]
                    display_cols = [col.replace('_outlier', '') for col in outlier_cols]
                    cols_to_remove = st.multiselect(
                        "Select columns to remove outliers from",
                        options=display_cols
                    )
                
                if st.button("Apply Outlier Removal", type="secondary"):
                    if removal_option == "Remove all detected outliers":
                        mask = ~st.session_state.outliers['is_outlier']
                        before_rows = len(st.session_state.cleaned_df)
                        st.session_state.cleaned_df = st.session_state.cleaned_df[mask]
                        after_rows = len(st.session_state.cleaned_df)
                        st.success(f"Removed {before_rows - after_rows} rows containing outliers")
                    
                    elif removal_option == "Remove outliers from specific columns" and 'cols_to_remove' in locals():
                        mask = pd.Series(True, index=st.session_state.cleaned_df.index)
                        for col in cols_to_remove:
                            outlier_col = f'{col}_outlier'
                            if outlier_col in st.session_state.outliers.columns:
                                mask = mask & ~st.session_state.outliers[outlier_col]
                        
                        before_rows = len(st.session_state.cleaned_df)
                        st.session_state.cleaned_df = st.session_state.cleaned_df[mask]
                        after_rows = len(st.session_state.cleaned_df)
                        st.success(f"Removed {before_rows - after_rows} rows containing outliers in selected columns")
                    
                    # Clear outlier detection results after removal
                    if 'outliers' in st.session_state:
                        del st.session_state.outliers
                    if 'outlier_details' in st.session_state:
                        del st.session_state.outlier_details
                    st.rerun()

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
st.caption("💡 Tip: Use the Data Cleaning and Outlier Detection tabs to prepare your data for model training!")