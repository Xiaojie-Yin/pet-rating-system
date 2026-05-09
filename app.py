import os
import random
import csv
from datetime import datetime
import re
import zipfile
import time

import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import requests


# ===============================
# Config (EDIT HERE)
# ===============================

# --- Online data root (after unzip) ---
DATA_ROOT = "data/dataset"

# Hugging Face zip URL
# The provided URL is kept unchanged
DATA_ZIP_URL = "https://huggingface.co/datasets/jxyz1224/pet-rating-data/resolve/main/dataset.zip?download=true"
DATA_ZIP_PATH = "data/dataset.zip"

# Maximum number of cases to evaluate (None = all cases)
MAX_CASES = 50

# Displayed total number of cases in the sidebar.
# This is fixed for presentation/screenshot purposes and does not depend on
# how many valid cases are actually found in DATA_ROOT.
DISPLAY_TOTAL_CASES = 50

# Save directory (results/ under the project root is convenient for later download)
SAVE_DIR = "results"
SAVE_FILE = os.path.join(SAVE_DIR, "ratings.csv")

# Window (fixed)
CT_MIN, CT_MAX = -160, 240
PET_MIN, PET_MAX = -2, 18  # display window [PET_MIN, PET_MAX]

# Slice rules
DROP_FIRST_LAST_SLICE = True  # Remove the first and last slices
AUTO_INIT_SLICE = True        # Initialize using the slice with the maximum SUVmax(A) + SUVmax(B)

# Download lock (avoid multiple downloads)
DOWNLOAD_LOCK = "data/.download.lock"
DOWNLOAD_DONE = "data/.download.done"


# ===============================
# Data preparation (download & unzip)
# ===============================

def ensure_data_ready():
    """
    Ensure DATA_ROOT exists and contains data.
    First run:
      - download DATA_ZIP_URL -> DATA_ZIP_PATH
      - unzip to data/
      - expect resulting folder: data/dataset/...
    """
    os.makedirs("data", exist_ok=True)

    # Already prepared
    if os.path.exists(DOWNLOAD_DONE) and os.path.exists(DATA_ROOT) and len(os.listdir(DATA_ROOT)) > 0:
        return

    # If another session is downloading, wait
    if os.path.exists(DOWNLOAD_LOCK) and not os.path.exists(DOWNLOAD_DONE):
        with st.spinner("Data are being prepared by another session. Please wait..."):
            # Wait up to ~10 minutes
            for _ in range(600):
                if os.path.exists(DOWNLOAD_DONE) and os.path.exists(DATA_ROOT) and len(os.listdir(DATA_ROOT)) > 0:
                    return
                time.sleep(1)
        # If still not ready, continue to attempt ourselves (lock may be stale)

    # Acquire lock
    try:
        with open(DOWNLOAD_LOCK, "w", encoding="utf-8") as f:
            f.write(str(datetime.now()))
    except Exception:
        pass

    try:
        # If folder exists but empty, still download
        st.warning("The rating dataset needs to be downloaded on first launch (approximately 400 MB). Please wait...")

        # Download zip
        with st.spinner("Downloading data..."):
            r = requests.get(DATA_ZIP_URL, stream=True, timeout=600)
            r.raise_for_status()
            with open(DATA_ZIP_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        # Unzip
        with st.spinner("Extracting data..."):
            with zipfile.ZipFile(DATA_ZIP_PATH, "r") as zf:
                zf.extractall("data")

        # Validate
        if not (os.path.exists(DATA_ROOT) and len(os.listdir(DATA_ROOT)) > 0):
            st.error(
                "No valid DATA_ROOT was found after extraction.\n\n"
                f"Expected path: {DATA_ROOT}\n"
                "Please check whether the zip archive contains a dataset/ folder."
            )
            st.stop()

        # Mark done
        with open(DOWNLOAD_DONE, "w", encoding="utf-8") as f:
            f.write(str(datetime.now()))

        st.success("Data are ready.")

    except requests.HTTPError as e:
        st.error(
            "Failed to download the dataset. The Hugging Face dataset may be private or inaccessible.\n\n"
            f"HTTPError: {e}\n\n"
            "Possible solutions:\n"
            "1) Make the Hugging Face dataset public; or\n"
            "2) Set HF_TOKEN in Streamlit Cloud and retry.\n"
        )
        st.stop()
    except Exception as e:
        st.error(f"Data preparation failed: {repr(e)}")
        st.stop()
    finally:
        # Remove lock (best effort)
        try:
            if os.path.exists(DOWNLOAD_LOCK):
                os.remove(DOWNLOAD_LOCK)
        except Exception:
            pass


# ===============================
# Utils
# ===============================

def load_nii(path: str) -> np.ndarray:
    nii = nib.load(path)
    return nii.get_fdata(dtype=np.float32)


def rotate_clockwise_90(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=-1)


def find_file(folder, keywords):
    """
    Find a file in folder that contains all keywords (case-insensitive)
    keywords: list[str]
    """
    for f in os.listdir(folder):
        f_low = f.lower()
        if all(k in f_low for k in keywords):
            return os.path.join(folder, f)
    return None


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


def prepare_cases(root, max_cases=None):
    """
    Support arbitrary patient ID folder names
    Automatically search CT / PET1 / PET2
    Keep cases in natural folder order; only randomize PET A/B
    """
    cases = []

    if not os.path.exists(root):
        st.error(f"DATA_ROOT not found: {root}")
        return cases

    all_folders = sorted(os.listdir(root), key=natural_key)

    for name in all_folders:
        case_dir = os.path.join(root, name)
        if not os.path.isdir(case_dir):
            continue

        ct = find_file(case_dir, ["ct", ".nii"])
        pt1 = find_file(case_dir, ["pet1", ".nii"])
        pt2 = find_file(case_dir, ["pet2", ".nii"])

        if ct is None or pt1 is None or pt2 is None:
            continue

        # Randomize A/B only; do not shuffle case order
        if random.random() < 0.5:
            a, b = pt1, pt2
            gt = "A"
        else:
            a, b = pt2, pt1
            gt = "B"

        cases.append({
            "id": name,
            "ct": ct,
            "A": a,
            "B": b,
            "gt": gt,  # hidden: which is real
        })

    if max_cases is not None:
        cases = cases[:max_cases]

    return cases


def save_rating(row):
    """
    CSV columns:
      Timestamp, Reviewer, CaseID,
      QualityA, QualityB,
      ContrastA, ContrastB,
      GuessGT,
      HiddenGT, PathA, PathB
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    file_exists = os.path.exists(SAVE_FILE)

    with open(SAVE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Reviewer",
                "CaseID",
                "QualityA",
                "QualityB",
                "ContrastA",
                "ContrastB",
                "GuessGT",
                "HiddenGT",
                "PathA",
                "PathB",
            ])

        writer.writerow([
            row["timestamp"],
            row["reviewer"],
            row["case_id"],
            row["quality_a"],
            row["quality_b"],
            row["contrast_a"],
            row["contrast_b"],
            row["guess_gt"],
            row["hidden_gt"],
            row["path_a"],
            row["path_b"],
        ])


def compute_valid_z_range(z_max: int):
    """
    If dropping first/last slice:
      valid z indices = [1, z_max-1]
    otherwise:
      valid z indices = [0, z_max]
    Return (z_min, z_max_valid)
    """
    if DROP_FIRST_LAST_SLICE and z_max >= 2:
        return 1, z_max - 1
    return 0, z_max


def suggest_initial_slice(a_vol: np.ndarray, b_vol: np.ndarray, z_min: int, z_max_valid: int) -> int:
    """
    Find z that maximizes: max(A[:,:,z]) + max(B[:,:,z])
    within [z_min, z_max_valid].
    """
    if not AUTO_INIT_SLICE:
        return (z_min + z_max_valid) // 2

    best_z = (z_min + z_max_valid) // 2
    best_score = -1e18

    for z in range(z_min, z_max_valid + 1):
        score = float(np.max(a_vol[:, :, z]) + np.max(b_vol[:, :, z]))
        if score > best_score:
            best_score = score
            best_z = z

    return best_z


# ===============================
# Init
# ===============================

st.set_page_config(layout="wide")

# Slightly widen the sidebar and keep the main content visually compact.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 360px;
        max-width: 360px;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 360px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Blinded PET Rating System")

# --------- Instruction "modal" (first open) ----------
if "show_instructions" not in st.session_state:
    st.session_state.show_instructions = True

if st.session_state.show_instructions:

    st.info(
        "### Instructions\n\n"
        "1）Please enter your name in the sidebar first.\n\n"
        "2) The system will sequentially display the CT image and two PET images (A / B) for each patient,"
        " with the initial slice automatically set near the level with the most prominent metabolic uptake.\n\n"
        "3) Use the slice slider below to review different axial slices, then rate PET A and PET B separately:\n"
        "   - Image quality (1–5)\n"
        "   - Tumor contrast (1–5)\n\n"
        "4) Indicate which PET image is more likely to be the real PET (A or B).\n\n"
        "5) After clicking “Submit and Continue”, the result will be saved automatically and the next case will be shown.\n\n"
        "6) After all cases are completed, click the “Download Results” button at the bottom of the page to download the rating table.\n\n"
        "Note: all images are displayed using fixed window settings, without automatic contrast adjustment."
    )

    if st.button("I understand. Start rating", type="primary"):
        st.session_state.show_instructions = False
        st.rerun()

    st.stop()


# Ensure data exists (download & unzip if needed)
ensure_data_ready()

# Initialize only once per session
if "initialized" not in st.session_state:
    st.session_state.cases = prepare_cases(DATA_ROOT, max_cases=MAX_CASES)
    st.session_state.idx = 0
    st.session_state.initialized = True

cases = st.session_state.cases
idx = st.session_state.idx


# ===============================
# Sidebar
# ===============================

st.sidebar.header("Reviewer Info")
reviewer = st.sidebar.text_input("Your Name", "")

st.sidebar.markdown("---")
st.sidebar.write("Dataset root:")
st.sidebar.code(DATA_ROOT)

st.sidebar.write("Save to:")
st.sidebar.code(SAVE_FILE)

st.sidebar.markdown("---")
st.sidebar.write(f"Case: {min(idx + 1, DISPLAY_TOTAL_CASES)} / {DISPLAY_TOTAL_CASES}")
st.sidebar.write(f"Max cases: {DISPLAY_TOTAL_CASES}")
st.sidebar.write(f"Auto initial slice: {AUTO_INIT_SLICE}")

with st.sidebar.expander("Admin", expanded=False):

    if st.button("Reset session"):
        st.session_state.cases = prepare_cases(
            DATA_ROOT,
            max_cases=MAX_CASES
        )
        st.session_state.idx = 0
        st.session_state.show_instructions = True
        st.rerun()

    st.markdown("---")
    st.subheader("Download Results")

    if os.path.exists(SAVE_FILE):

        with open(SAVE_FILE, "rb") as f:
            st.download_button(
                label="Download ratings.csv",
                data=f,
                file_name="ratings.csv",
                mime="text/csv"
            )
    else:
        st.info("No results file yet.")


# ===============================
# Main
# ===============================

if len(cases) == 0:
    st.error(
        "No valid cases found.\n\n"
        "Each case folder must contain:\n"
        "  CT*.nii(.gz)\n"
        "  PET1*.nii(.gz)\n"
        "  PET2*.nii(.gz)\n\n"
        f"DATA_ROOT = {DATA_ROOT}"
    )
    st.stop()

if st.session_state.get("finished", False):
    st.success("🎉 All cases have been completed. Thank you for your participation!")
    st.balloons()

    st.markdown("### 📄 Download Results")

    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "rb") as f:
                st.download_button(
                    label="⬇️ Download Results (ratings.csv)",
                    data=f,
                    file_name="ratings.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Failed to read the result file: {repr(e)}")
    else:
        st.error("The result file (ratings.csv) was not found. Please contact the administrator to confirm the save path or check whether any ratings have been submitted.")

    st.stop()


case = cases[idx]


# ===============================
# Load data
# ===============================

@st.cache_data(show_spinner=False)
def load_case(ct, a, b):
    return (load_nii(ct), load_nii(a), load_nii(b))

ct_vol, a_vol, b_vol = load_case(case["ct"], case["A"], case["B"])

# assume (H, W, Z)
z_max = ct_vol.shape[2] - 1
z_min, z_max_valid = compute_valid_z_range(z_max)

# initial slice
init_z = suggest_initial_slice(a_vol, b_vol, z_min, z_max_valid)


# ===============================
# Slice
# ===============================

z = st.slider(
    "Slice (Axial)",
    min_value=z_min,
    max_value=z_max_valid,
    value=init_z
)

ct_slice = ct_vol[:, :, z]
a_slice = a_vol[:, :, z]
b_slice = b_vol[:, :, z]

ct_disp = rotate_clockwise_90(ct_slice)
a_disp = rotate_clockwise_90(a_slice)
b_disp = rotate_clockwise_90(b_slice)


# ===============================
# Visualization (fixed window/level)
# ===============================

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))

axes[0].imshow(ct_disp, cmap="gray", vmin=CT_MIN, vmax=CT_MAX)
axes[0].set_title(f"CT (HU [{CT_MIN}, {CT_MAX}])")
axes[0].axis("off")

axes[1].imshow(a_disp, cmap="gray", vmin=PET_MIN, vmax=PET_MAX)
axes[1].set_title(f"PET A (SUV [{PET_MIN + 2}, {PET_MAX + 2}])")
axes[1].axis("off")

axes[2].imshow(b_disp, cmap="gray", vmin=PET_MIN, vmax=PET_MAX)
axes[2].set_title(f"PET B (SUV [{PET_MIN + 2}, {PET_MAX + 2}])")
axes[2].axis("off")

fig.tight_layout()
st.pyplot(fig, use_container_width=False)


# ===============================
# Rating
# ===============================

st.markdown("### Rating (1–5; higher scores indicate better quality)")

st.caption("Please rate PET A and PET B separately.")

c1, c2 = st.columns(2)

with c1:
    st.subheader("PET A")
    quality_a = st.slider("Image quality (A)", 1, 5, 3, key=f"qa_{idx}")
    contrast_a = st.slider("Tumor contrast (A)", 1, 5, 3, key=f"ca_{idx}")

with c2:
    st.subheader("PET B")
    quality_b = st.slider("Image quality (B)", 1, 5, 3, key=f"qb_{idx}")
    contrast_b = st.slider("Tumor contrast (B)", 1, 5, 3, key=f"cb_{idx}")

st.markdown("### Which image is more likely to be the real PET?")
guess_gt = st.radio("Please select", ["A", "B"], horizontal=True, key=f"guess_{idx}")

submit = st.button("Submit and Continue", type="primary")


# ===============================
# Submit
# ===============================

if submit:
    if reviewer.strip() == "":
        st.error("Please enter your name in the sidebar first.")
        st.stop()

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reviewer": reviewer.strip(),
        "case_id": case["id"],
        "quality_a": int(quality_a),
        "quality_b": int(quality_b),
        "contrast_a": int(contrast_a),
        "contrast_b": int(contrast_b),
        "guess_gt": guess_gt,
        "hidden_gt": case["gt"],
        "path_a": case["A"],
        "path_b": case["B"],
    }

    # ---- save first ----
    try:
        save_rating(record)
    except Exception as e:
        st.error(f"Failed to save the rating: {repr(e)}")
        st.stop()

    # ---- update index ----
    st.session_state.idx += 1

    # ---- if finished, go to done page ----
    if st.session_state.idx >= len(cases):
        st.session_state.finished = True
        st.rerun()

    # ---- otherwise continue ----
    st.rerun()








