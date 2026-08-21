import re
import streamlit as st
import pandas as pd
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, rdinchi
import uuid

st.set_page_config(page_title="MS/MS Session Builder", layout="wide")

# --- Monoisotopic masses for formula-only entries (no structure available) ---
MONO_MASSES = {
    'H': 1.00782503207, 'D': 2.01410177785, 'C': 12.0, 'N': 14.0030740048,
    'O': 15.9949146196, 'F': 18.99840322, 'Na': 22.9897692809, 'Mg': 23.9850417,
    'Al': 26.98153863, 'Si': 27.9769265325, 'P': 30.97376163, 'S': 31.97207100,
    'Cl': 34.96885268, 'K': 38.9637069, 'Ca': 39.962591, 'Mn': 54.9380451,
    'Fe': 55.9349375, 'Co': 58.9331950, 'Ni': 57.9353429, 'Cu': 62.9295975,
    'Zn': 63.9291422, 'Br': 78.9183371, 'Se': 79.9165213, 'I': 126.904473,
    'B': 11.0093054,
}
ELECTRON_MASS = 0.00054858

# --- Initialize Session State ---
if "session_data" not in st.session_state:
    st.session_state.session_data = []

if "inchi_input" not in st.session_state:
    st.session_state.inchi_input = ""

if "formula_input" not in st.session_state:
    st.session_state.formula_input = ""

if "entry_type" not in st.session_state:
    st.session_state.entry_type = "Fragment Ion"

if "feedback" not in st.session_state:
    st.session_state.feedback = None

if "last_added_img" not in st.session_state:
    st.session_state.last_added_img = None

if "last_added_caption" not in st.session_state:
    st.session_state.last_added_caption = None


# --- RDKit Processing Function (InChI -> SMILES / formula / mass) ---
from rdkit.Chem import rdinchi # Add this to your imports at the top

def process_inchi(inchi_string):
    mol = Chem.MolFromInchi(inchi_string)
    
    if mol is None:
        # Fall back to rdinchi to extract the diagnostic error message
        try:
            _, retcode, message = rdinchi.InchiToMol(inchi_string)
            error_msg = message if message else f"Unparseable syntax or invalid valency (Return code: {retcode})"
        except Exception as e:
            error_msg = str(e)
        return None, None, error_msg

    img = Draw.MolToImage(mol, size=(300, 300))
    data = {
        'smiles': Chem.MolToSmiles(mol),
        'inchi': Chem.MolToInchi(mol),
        'inchikey': Chem.MolToInchiKey(mol),
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'exact_mass': round(rdMolDescriptors.CalcExactMolWt(mol), 4)
    }
    return data, img, None

# --- Formula-only mass calculation (no structure available) ---
def parse_formula_mass(formula_str):
    """Parses a molecular formula (optionally with a trailing charge, e.g. 'C6H5+')
    and returns (exact_mass, clean_formula). Returns (None, None) on failure."""
    formula_str = formula_str.strip().replace(" ", "")
    if not formula_str:
        return None, None

    # Pull off a trailing charge notation, e.g. "+", "-", "2+", "2-"
    charge = 0
    charge_match = re.search(r'(\d*)([+-])$', formula_str)
    body = formula_str
    if charge_match:
        num = charge_match.group(1)
        sign = charge_match.group(2)
        magnitude = int(num) if num else 1
        charge = magnitude if sign == '+' else -magnitude
        body = formula_str[:charge_match.start()]

    atom_pattern = re.findall(r'([A-Z][a-z]?)(\d*)', body)
    total_mass = 0.0
    clean_parts = []
    matched_any_chars = 0

    for element, count_str in atom_pattern:
        if not element:
            continue
        matched_any_chars += len(element) + len(count_str)
        count = int(count_str) if count_str else 1
        if element not in MONO_MASSES:
            return None, None  # unknown / unsupported element symbol
        total_mass += MONO_MASSES[element] * count
        clean_parts.append(f"{element}{count if count > 1 else ''}")

    # Make sure the whole body was consumed by the regex (catches typos/garbage input)
    if matched_any_chars != len(body) or not clean_parts:
        return None, None

    # Correct for missing/extra electrons if the formula represents a charged ion
    total_mass -= charge * ELECTRON_MASS

    clean_formula = "".join(clean_parts) + (charge_match.group(0) if charge_match else "")
    return round(total_mass, 4), clean_formula


# --- Dialog Popup for Structure Preview ---
@st.dialog("Chemical Structure Preview")
def preview_structure_dialog(inchi_str):
    inchi_clean = inchi_str.strip()
    if not inchi_clean:
        st.warning("Please paste or enter an InChI string first.")
        return

    chem_data, mol_img, err_msg = process_inchi(inchi_clean)

    if chem_data and mol_img:
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(mol_img, caption="2D Structure Render", use_container_width=True)
        with col_info:
            st.markdown("### Molecular Properties")
            st.metric("Formula", chem_data['formula'])
            st.metric("Exact Mass", f"{chem_data['exact_mass']:.4f} Da")
            st.markdown("**InChIKey:**")
            st.code(chem_data['inchikey'], language=None)
            st.markdown("**Canonical SMILES:**")
            st.code(chem_data['smiles'], language=None)
    else:
        st.error("Invalid InChI string. RDKit could not parse the structure.")
        st.info(f"**Diagnostic Info:**\n{err_msg}")


# --- Callbacks ---

def set_entry_type(entry_type):
    st.session_state.entry_type = entry_type


def build_metadata_fields():
    """Collects the sidebar metadata that gets copied onto every table row."""
    return {
        'parent_ion_mz': st.session_state.get("meta_parent_ion", ""),
        'adduct': st.session_state.get("meta_adduct", ""),
        'ionization_mode': st.session_state.get("meta_ionization", ""),
        'ion_source': st.session_state.get("meta_source", ""),
        'instrument': st.session_state.get("meta_instrument", ""),
        'doi': st.session_state.get("meta_doi", ""),
        'mechanism_in_reference': "Yes" if st.session_state.get("meta_mechanism", False) else "No",
        'compound_class': st.session_state.get("meta_compound_class", ""),
    }


def add_structure():
    inchi_val = st.session_state.inchi_input.strip()
    if not inchi_val:
        st.session_state.feedback = ("warning", "Please paste or enter an InChI string first.")
        return

    chem_data, mol_img, err_msg = process_inchi(inchi_val)

    if chem_data:
        entry_type = st.session_state.entry_type
        new_record = {
            'id': str(uuid.uuid4())[:8],  # Generates a unique 8-character ID
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': entry_type,
            'formula': chem_data['formula'],
            'exact_mass': chem_data['exact_mass'],
            'smiles': chem_data['smiles'],
            'inchi': chem_data['inchi'],
            'inchikey': chem_data['inchikey'],
            **build_metadata_fields(),
        }

        if not any(r.get('inchikey') == chem_data['inchikey'] and r['type'] == entry_type
                   for r in st.session_state.session_data):
            st.session_state.session_data.append(new_record)
            st.session_state.feedback = (
                "toast",
                f"Added {entry_type}: {chem_data['formula']} (m/z {chem_data['exact_mass']:.4f})"
            )
            st.session_state.inchi_input = ""
            st.session_state.last_added_img = mol_img
            st.session_state.last_added_caption = (
                f"{entry_type} — {chem_data['formula']} — Exact Mass: {chem_data['exact_mass']:.4f} Da"
            )
        else:
            st.session_state.feedback = ("warning", "This structure and entry type combination is already in your current session.")
    else:
        # Show diagnostic error on failure
        st.session_state.feedback = ("error", f"Invalid InChI string.\nDiagnostic: {err_msg}")


def add_formula_entry():
    formula_val = st.session_state.formula_input.strip()
    if not formula_val:
        st.session_state.feedback = ("warning", "Please paste or enter a molecular formula first.")
        return

    mass, clean_formula = parse_formula_mass(formula_val)

    if mass is None:
        st.session_state.feedback = (
            "error",
            "Could not parse that formula. Use standard notation, e.g. C6H12O6 or C6H5+."
        )
        return

    entry_type = st.session_state.entry_type
    new_record = {
        'id': str(uuid.uuid4())[:8],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'type': entry_type,
        'formula': clean_formula,
        'exact_mass': mass,
        'smiles': 'NA',
        'inchi': 'NA',
        'inchikey': 'NA',
        **build_metadata_fields(),
    }

    if not any(r.get('formula') == clean_formula and r['type'] == entry_type and r.get('smiles') == 'NA'
               for r in st.session_state.session_data):
        st.session_state.session_data.append(new_record)
        st.session_state.feedback = ("toast", f"Added {entry_type} (formula only): {clean_formula} (mass {mass:.4f})")
        st.session_state.formula_input = ""
        st.session_state.last_added_img = None
        st.session_state.last_added_caption = f"{entry_type} — {clean_formula} — Exact Mass: {mass:.4f} Da (no structure available)"
    else:
        st.session_state.feedback = ("warning", "This formula and entry type combination is already in your current session.")


# --- Sidebar Metadata ---
with st.sidebar:
    st.header("Global Metadata")
    st.caption("Applies automatically to every structure or formula you add.")

    st.subheader("Precursor Info")
    st.text_input("Parent Ion m/z", key="meta_parent_ion")
    st.text_input("Adduct", value="[M+H]+", key="meta_adduct")
    st.selectbox("Ionization Mode", ["Positive", "Negative"], key="meta_ionization")

    st.subheader("Instrumentation")
    st.text_input("Ion Source", value="ESI", key="meta_source")
    st.text_input("Instrument", value="Q-TOF", key="meta_instrument")

    st.subheader("Reference")
    st.text_input("Reference DOI", key="meta_doi")
    st.toggle("Mechanism Included in Reference", key="meta_mechanism")
    st.text_input("Compound Class", key="meta_compound_class")

    st.divider()

    uploaded_file = st.file_uploader("Resume from past CSV", type=["csv"])
    if uploaded_file is not None:
        if st.button("Load CSV into Session"):
            prev_df = pd.read_csv(uploaded_file)
            st.session_state.session_data = prev_df.to_dict('records')
            st.success(f"Loaded {len(prev_df)} records!")
            st.rerun()

# --- Main Interface ---
st.title("MS/MS Fragment & Neutral Loss Library")

# Entry Type selector (Fragment Ion / Neutral Loss)
st.write("**Select Entry Type:**")
col_type_frag, col_type_nl = st.columns(2)
with col_type_frag:
    st.button(
        "Fragment Ion",
        on_click=set_entry_type, args=("Fragment Ion",),
        type="primary" if st.session_state.entry_type == "Fragment Ion" else "secondary",
        use_container_width=True
    )
with col_type_nl:
    st.button(
        "Neutral Loss",
        on_click=set_entry_type, args=("Neutral Loss",),
        type="primary" if st.session_state.entry_type == "Neutral Loss" else "secondary",
        use_container_width=True
    )
st.caption(f"Currently adding as: **{st.session_state.entry_type}**")

# InChI Input & Action Buttons (Row 1)
col_input, col_add = st.columns([4.5, 1.5])

with col_input:
    st.text_input(
        "InChI String:",
        key="inchi_input",
        placeholder="InChI=1S/..."
    )

with col_add:
    st.write("")
    st.write("")
    st.button("➕ Add Structure", type="primary", on_click=add_structure, use_container_width=True)

with col_add:
    st.write("")
    st.write("")
    st.button("➕ Add Structure", type="primary", on_click=add_structure, use_container_width=True)

# Check Structure Row (Row 2)
col_check, _ = st.columns([1.5, 4.4])
with col_check:
    if st.button("🔍 Check Structure", use_container_width=True):
        preview_structure_dialog(st.session_state.inchi_input)

# Formula-only entry (no structure available)
# Formula-only entry (no structure available)
with st.expander("➕ Add a formula-only entry (no structure available)"):
    col_f_input, col_f_add = st.columns([4.5, 1.5])
    
    with col_f_input:
        st.text_input("Molecular Formula:", key="formula_input", placeholder="e.g. C6H12O6 or C6H5+")
        
    with col_f_add:
        st.write("")
        st.write("")
        st.button("➕ Add Formula", on_click=add_formula_entry, use_container_width=True, key="add_formula_btn")
        
    st.caption("Structure fields (SMILES / InChI / InChIKey) will be stored as \"NA\" for these entries.")

# Display feedback messages triggered by callbacks
if st.session_state.feedback:
    fb_type, fb_msg = st.session_state.feedback
    if fb_type == "toast":
        st.toast(fb_msg, icon="✅")
    elif fb_type == "warning":
        st.warning(fb_msg)
    elif fb_type == "error":
        st.error(fb_msg)
    st.session_state.feedback = None  # Reset feedback

# Persistent preview of the last structure that was added, for a quick sanity check
if st.session_state.last_added_caption:
    st.divider()
    col_prev_img, col_prev_txt = st.columns([1, 4])
    with col_prev_img:
        if st.session_state.last_added_img is not None:
            st.image(st.session_state.last_added_img, width=150)
        else:
            st.info("No structure\n(formula only)")
    with col_prev_txt:
        st.success(f"**Last added:** {st.session_state.last_added_caption}")

# --- View All Structures Gallery ---
if st.session_state.session_data:
    with st.expander("🖼️ View All Structures in Session"):
        # Filter out formula-only entries (where inchi is 'NA')
        structure_entries = [row for row in st.session_state.session_data if row.get('inchi') != 'NA']
        
        if not structure_entries:
            st.info("No structural data to display (only formula entries are currently in the session).")
        else:
            # Create a 4-column grid
            cols = st.columns(4)
            for index, row in enumerate(structure_entries):
                mol = Chem.MolFromInchi(row['inchi'])
                if mol:
                    # Generate the image on the fly
                    img = Draw.MolToImage(mol, size=(800, 800))
                    
                    # Distribute across columns based on index
                    with cols[index % 4]:
                        # Combine ID, type, and formula in the caption
                        caption_text = f"**ID:** {row.get('id', 'N/A')}\n\n**{row.get('type', '')}**\n\n{row.get('formula', '')}"
                        st.image(img, use_container_width=True)
                        st.markdown(caption_text)

# --- Session Table Display & Actions ---
st.subheader("Current Session Entries")

if st.session_state.session_data:
    df_session = pd.DataFrame(st.session_state.session_data)

    edited_df = st.data_editor(
        df_session,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True
    )
    st.session_state.session_data = edited_df.to_dict('records')

    st.divider()

    col_dl, col_clear = st.columns([3, 1])
    csv_filename = f"msms_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_bytes = edited_df.to_csv(index=False).encode('utf-8')

    with col_dl:
        st.download_button(
            label="⬇️ Download Session CSV",
            data=csv_bytes,
            file_name=csv_filename,
            mime="text/csv",
            type="primary"
        )

    with col_clear:
        if st.button("Clear Current Session"):
            st.session_state.session_data = []
            st.session_state.last_added_img = None
            st.session_state.last_added_caption = None
            st.rerun()
else:
    st.info("No entries yet in this session. Select an entry type, paste an InChI (or formula), and add it to get started.")
