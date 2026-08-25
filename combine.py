import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="MS/MS CSV Integrator", page_icon="🧪", layout="wide")

st.title("🧪 MS/MS Session CSV Integrator")
st.markdown("Upload individual session CSVs or previously combined master files to merge and expand your dataset.")

# File uploader accepting multiple files
uploaded_files = st.file_uploader(
    "Select MS/MS Session CSV files",
    type=["csv"],
    accept_multiple_files=True,
    help="You can upload multiple individual session files along with any previously merged master CSV."
)

if uploaded_files:
    dfs = []
    file_summary = []

    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            file_summary.append({"Filename": file.name, "Rows": len(df)})
        except Exception as e:
            st.error(f"Error reading `{file.name}`: {e}")

    if dfs:
        # Merge all dataframes into one
        combined_df = pd.concat(dfs, ignore_index=True)

        # Sidebar Controls
        st.sidebar.header("Options")
        drop_dupes = st.sidebar.checkbox("Remove exact duplicate rows", value=True)
        
        initial_row_count = len(combined_df)
        if drop_dupes:
            combined_df = combined_df.drop_duplicates()
        final_row_count = len(combined_df)
        dupes_removed = initial_row_count - final_row_count

        # Summary Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Files Uploaded", len(uploaded_files))
        m2.metric("Total Combined Rows", final_row_count)
        m3.metric("Duplicates Removed", dupes_removed)

        # Detailed view of uploaded sources
        with st.expander("📁 Uploaded Files Summary"):
            st.table(pd.DataFrame(file_summary))

        st.divider()

        # Data Editor / Interactive Table
        st.subheader("Combined Dataset Preview")
        edited_combined_df = st.data_editor(
            combined_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="combined_editor"
        )

        st.divider()

        # Export Section
        col_input, col_dl = st.columns([3, 1])

        with col_input:
            custom_label = st.text_input(
                "Master Filename Label (optional)",
                placeholder="e.g., phenolics_master_dataset",
                key="master_name_input"
            )

        with col_dl:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if custom_label.strip():
                clean_name = custom_label.strip().replace(" ", "_").removesuffix(".csv")
                filename = f"msms_master_{clean_name}_{timestamp}.csv"
            else:
                filename = f"msms_master_{timestamp}.csv"

            csv_bytes = edited_combined_df.to_csv(index=False, quoting=1).encode('utf-8')

            st.write("##") # Spacing alignment with text input
            st.download_button(
                label="⬇️ Download Master CSV",
                data=csv_bytes,
                file_name=filename,
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
else:
    st.info("Upload one or more CSV files (e.g., `msms_session_*.csv` or a prior `msms_master_*.csv`) to generate an integrated export.")
