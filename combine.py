import streamlit as st
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="MS/MS Database Accumulator", page_icon="🗃️", layout="wide")

DB_FILE = "msms_master_database.csv"

# Initialize Session State Database
if "master_df" not in st.session_state:
    if os.path.exists(DB_FILE):
        st.session_state.master_df = pd.read_csv(DB_FILE)
    else:
        st.session_state.master_df = pd.DataFrame()

st.title("🗃️ MS/MS Master Database Accumulator")

# --- Step 1: Base Database Status & Initial Import ---
st.subheader("1. Active Master Database")

col_info, col_import = st.columns([2, 2])

with col_info:
    st.metric("Total Records in Database", len(st.session_state.master_df))
    if os.path.exists(DB_FILE):
        st.caption(f"📁 Auto-loaded local storage file: `{DB_FILE}`")

with col_import:
    with st.expander("📥 Import External Master CSV (Optional)"):
        master_upload = st.file_uploader(
            "Upload an existing master file (e.g., sent by a colleague)",
            type=["csv"],
            key="master_uploader"
        )
        if master_upload and st.button("Overwrite / Load as Current Master"):
            try:
                st.session_state.master_df = pd.read_csv(master_upload)
                st.session_state.master_df.to_csv(DB_FILE, index=False)
                st.success("Master database updated from file!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading master CSV: {e}")

st.divider()

# --- Step 2: Append New Daily/Session CSVs ---
st.subheader("2. Append New Session CSVs")

new_files = st.file_uploader(
    "Select new session CSVs to add",
    type=["csv"],
    accept_multiple_files=True,
    key="new_sessions_uploader"
)

if new_files:
    if st.button("⚡ Merge & Append New Files to Master", type="primary"):
        new_dfs = []
        for file in new_files:
            try:
                df = pd.read_csv(file)
                new_dfs.append(df)
            except Exception as e:
                st.error(f"Error reading `{file.name}`: {e}")

        if new_dfs:
            # Combine current database with all newly uploaded session CSVs
            all_dfs = [st.session_state.master_df] + new_dfs
            merged_df = pd.concat(all_dfs, ignore_index=True)

            # Deduplicate across the entire combined dataset
            before_count = len(merged_df)
            merged_df = merged_df.drop_duplicates()
            after_count = len(merged_df)

            # Update session state & auto-save to local disk file
            st.session_state.master_df = merged_df
            st.session_state.master_df.to_csv(DB_FILE, index=False)

            st.success(f"Added {before_count - len(st.session_state.master_df)} new rows! ({before_count - after_count} exact duplicates filtered). Auto-saved to `{DB_FILE}`.")
            st.rerun()

st.divider()

# --- Step 3: Interactive View & Export ---
st.subheader("3. Integrated Master Database View")

if not st.session_state.master_df.empty:
    st.session_state.master_df = st.data_editor(
        st.session_state.master_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="master_data_editor"
    )

    col_dl, col_reset = st.columns([3, 1])

    with col_dl:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        custom_name = st.text_input("Custom Label for Export (optional)", placeholder="e.g., lycium_project_v2")
        
        if custom_name.strip():
            clean_label = custom_name.strip().replace(" ", "_").removesuffix(".csv")
            export_filename = f"msms_master_{clean_label}_{timestamp}.csv"
        else:
            export_filename = f"msms_master_{timestamp}.csv"

        csv_bytes = st.session_state.master_df.to_csv(index=False, quoting=1).encode('utf-8')
        st.download_button(
            label="⬇️ Export Master CSV for Colleague",
            data=csv_bytes,
            file_name=export_filename,
            mime="text/csv"
        )

    with col_reset:
        st.write("##")
        if st.button("🗑️ Reset Local Master DB"):
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
            st.session_state.master_df = pd.DataFrame()
            st.rerun()
else:
    st.info("The master database is currently empty. Upload a master CSV or add new session CSVs above.")
