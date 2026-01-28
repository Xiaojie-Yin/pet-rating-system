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
# 你给的链接可用；我这里保留原样
DATA_ZIP_URL = "https://huggingface.co/datasets/jxyz1224/pet-rating-data/resolve/main/dataset.zip?download=true"
DATA_ZIP_PATH = "data/dataset.zip"

# 最多评估多少例（None = 全部）
MAX_CASES = 40

# 保存目录（建议放到项目根目录的 results/，便于你后面下载）
SAVE_DIR = "results"
SAVE_FILE = os.path.join(SAVE_DIR, "ratings.csv")

# Window (fixed)
CT_MIN, CT_MAX = -160, 240
PET_MIN, PET_MAX = -2, 18  # display window [PET_MIN, PET_MAX]

# Slice rules
DROP_FIRST_LAST_SLICE = True  # 去掉第1张和最后1张切片
AUTO_INIT_SLICE = True        # 初始切片用“(A的SUVmax + B的SUVmax) 最大”的切片

# Download lock (avoid multiple downloads)
DOWNLOAD_LOCK = "data/.download.lock"
DOWNLOAD_DONE = "data/.download.done"


# ===============================
# Data 준비 (download & unzip)
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
        with st.spinner("数据正在准备中（其他会话正在下载/解压），请稍候..."):
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
        st.warning("首次启动需要下载评估数据（约 600MB），请耐心等待...")

        # Download zip
        with st.spinner("正在下载数据..."):
            r = requests.get(DATA_ZIP_URL, stream=True, timeout=600)
            r.raise_for_status()
            with open(DATA_ZIP_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        # Unzip
        with st.spinner("正在解压数据..."):
            with zipfile.ZipFile(DATA_ZIP_PATH, "r") as zf:
                zf.extractall("data")

        # Validate
        if not (os.path.exists(DATA_ROOT) and len(os.listdir(DATA_ROOT)) > 0):
            st.error(
                "数据解压后未找到有效的 DATA_ROOT。\n\n"
                f"期望存在：{DATA_ROOT}\n"
                "请检查 zip 包内部是否包含 dataset/ 文件夹。"
            )
            st.stop()

        # Mark done
        with open(DOWNLOAD_DONE, "w", encoding="utf-8") as f:
            f.write(str(datetime.now()))

        st.success("数据已准备完成。")

    except requests.HTTPError as e:
        st.error(
            "下载数据失败（可能是 Hugging Face 数据集是 Private 导致无权限）。\n\n"
            f"HTTPError: {e}\n\n"
            "解决方案：\n"
            "1）把 Hugging Face 数据集改为 Public；或\n"
            "2）在 Streamlit Cloud 设置 HF_TOKEN 后再下载。\n"
        )
        st.stop()
    except Exception as e:
        st.error(f"数据准备失败：{repr(e)}")
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
    在 folder 里找包含 keywords 的文件（不区分大小写）
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
    支持任意患者ID文件夹名
    自动搜索 CT / PET1 / PET2
    保持文件夹自然顺序，不打乱case顺序；仅随机A/B
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

        # 只随机 A/B，不打乱 case 顺序
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
st.title("Blinded PET Rating System")

# --------- Instruction "modal" (first open) ----------
if "show_instructions" not in st.session_state:
    st.session_state.show_instructions = True

if st.session_state.show_instructions:
    st.info(
        "### 使用说明\n\n"
        "1）请先在左侧输入您的姓名。\n\n"
        "2）系统会依次显示每位患者的 CT 与两组 PET 图像（A / B），"
        "并自动定位在肿瘤代谢最明显的层面附近，便于评估。\n\n"
        "3）请通过下方滑块浏览不同切片后，对 PET A 和 PET B 分别进行评分：\n"
        "   - 成像质量（1–5 分）\n"
        "   - 肿瘤成像对比度（1–5 分）\n\n"
        "4）请判断哪一组 PET 更可能为真实 PET（A 或 B）。\n\n"
        "5）点击“提交并进入下一例”后，系统将自动保存结果并进入下一病例。\n\n"
        "说明：所有图像均采用固定窗宽窗位显示，不进行自动对比度调整。"
    )

    if st.button("我已了解，开始评估", type="primary"):
        st.session_state.show_instructions = False
        st.rerun()
    st.stop()

# Ensure data exists (download & unzip if needed)
ensure_data_ready()

# 初始化：只在首次进入会话时执行
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
st.sidebar.write(f"Case: {min(idx+1, max(len(cases), 1))} / {len(cases)}")
st.sidebar.write(f"Max cases: {MAX_CASES}")
st.sidebar.write(f"Auto initial slice: {AUTO_INIT_SLICE}")

with st.sidebar.expander("Admin", expanded=False):
    if st.button("Reset session"):
        st.session_state.cases = prepare_cases(DATA_ROOT, max_cases=MAX_CASES)
        st.session_state.idx = 0
        st.session_state.show_instructions = True
        st.rerun()


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

if idx >= len(cases):
    st.success("All cases completed. Thank you!")
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

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].imshow(ct_disp, cmap="gray", vmin=CT_MIN, vmax=CT_MAX)
axes[0].set_title(f"CT (HU [{CT_MIN}, {CT_MAX}])")
axes[0].axis("off")

axes[1].imshow(a_disp, cmap="gray", vmin=PET_MIN, vmax=PET_MAX)
axes[1].set_title(f"PET A (SUV [{PET_MIN}, {PET_MAX}])")
axes[1].axis("off")

axes[2].imshow(b_disp, cmap="gray", vmin=PET_MIN, vmax=PET_MAX)
axes[2].set_title(f"PET B (SUV [{PET_MIN}, {PET_MAX}])")
axes[2].axis("off")

st.pyplot(fig, use_container_width=True)


# ===============================
# Rating
# ===============================

st.markdown("### 评分（1–5 分，分数越高越好）")

st.caption("请分别对 PET A 和 PET B 进行评分。")

c1, c2 = st.columns(2)

with c1:
    st.subheader("PET A")
    quality_a = st.slider("成像质量（A）", 1, 5, 3, key=f"qa_{idx}")
    contrast_a = st.slider("肿瘤成像对比度（A）", 1, 5, 3, key=f"ca_{idx}")

with c2:
    st.subheader("PET B")
    quality_b = st.slider("成像质量（B）", 1, 5, 3, key=f"qb_{idx}")
    contrast_b = st.slider("肿瘤成像对比度（B）", 1, 5, 3, key=f"cb_{idx}")

st.markdown("### 您认为哪一个更可能是真实 PET？")
guess_gt = st.radio("请选择", ["A", "B"], horizontal=True, key=f"guess_{idx}")

submit = st.button("提交并进入下一例", type="primary")


# ===============================
# Submit
# ===============================

if submit:
    if reviewer.strip() == "":
        st.error("请先在左侧输入您的姓名。")
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

    save_rating(record)

    st.session_state.idx += 1
    st.rerun()
