import os
import sys
import numpy as np
import onnx
import qai_hub
from compile_and_profile import compile_model
import h5py

# =============================================================================
# USER CONFIGURATION — update these values before running
# =============================================================================

# Path to the directory containing your exported ONNX model.
ONNX_DIR = ""
VIDEO_ONNX_NAME = "model.onnx"
DEVICE_NAME = "Dragonwing IQ-9075 EVK"

# --- Pre-compiled model (optional) ---
# Set to the path of a pre-compiled .dlc or .bin file to skip the ONNX compile
# step and upload it directly to QAI Hub for inference.
# Leave empty ("") to use the ONNX → compile path instead.
DLC_PATH = ""  # e.g. "./export_assets/resnet_2plus1d.bin"

# Input dimensions — must match the model's expected input.
BATCH = 1
C = 3
T = 16
H = 112
W = 112

# Set to True if your DLC/binary was compiled with channel-last (NTHWC) input.
# The preprocessed .npy tensors are channel-first (NCTHW), so they will be
# transposed automatically when this is True.
# Set to False if your model uses channel-first (NCTHW) input (e.g. ONNX compile path).
IS_DLC_CHANNEL_LAST = False  # True for the official Qualcomm-provided .bin

# Path to the directory of preprocessed .npy tensors.
data_path = ""
# Name prefix used when uploading dataset chunks to QAI Hub.
DATASET_NAME = "dataset"

# Toggle single-tensor inference (useful for quick debugging) vs full-dataset inference.
USE_SINGLE_TENSOR = False
# Path to a specific .npy tensor to use when USE_SINGLE_TENSOR is True.
# Leave empty to automatically pick tensor number SINGLE_TENSOR_INDEX from data_path.
SINGLE_TENSOR_PATH = ""
SINGLE_TENSOR_INDEX = 0

# Output HDF5 file written after inference, consumed by evaluate.py.
OUTPUT_H5 = "dataset-export.h5"

# =============================================================================
# Early config validation — fail fast before any expensive work
# =============================================================================

_errors = []

if not data_path:
    _errors.append("'data_path' is not set. Point it to the directory of preprocessed .npy tensors.")
elif not os.path.isdir(data_path):
    _errors.append(f"'data_path' directory not found: '{data_path}'")

if DLC_PATH:
    # DLC path: only DLC_PATH needs to exist; ONNX_DIR is not used.
    if not os.path.exists(DLC_PATH):
        _errors.append(f"DLC_PATH file not found: '{DLC_PATH}'")
else:
    # ONNX path: ONNX_DIR must be set and the model file must exist.
    if not ONNX_DIR:
        _errors.append("'ONNX_DIR' is not set and 'DLC_PATH' is empty. "
                       "Either set DLC_PATH (pre-compiled model) or set ONNX_DIR (ONNX compile path).")
    else:
        _onnx_path = os.path.join(ONNX_DIR, VIDEO_ONNX_NAME)
        if not os.path.exists(_onnx_path):
            _errors.append(f"ONNX model not found: '{_onnx_path}'. "
                           "Set ONNX_DIR to the correct directory or run export first.")

if USE_SINGLE_TENSOR and not SINGLE_TENSOR_PATH and not data_path:
    _errors.append("USE_SINGLE_TENSOR is True but neither SINGLE_TENSOR_PATH nor data_path is set.")

if _errors:
    print("Configuration error(s) in run_inference.py — please fix before running:\n")
    for err in _errors:
        print(f"  ✗ {err}")
    sys.exit(1)

# =============================================================================

def inference_job(model, device, dataset):
    job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=dataset,
        options=""
    )
    return job.job_id

def _iter_npy_paths(root: str):
    """
    Yield .npy paths in the same order the manifest was written.

    IMPORTANT: preprocess_and_save.py calls list_videos() which collects all
    paths and does a single sorted() on the full path strings.  Sorting full
    paths means '-' (ASCII 45) < '/' (ASCII 47), so a class like
    'cross-legged_hamstring_stretch' sorts BEFORE 'cross' when compared as
    full paths.  A naive two-level sort (sort class dirs, then sort files
    within each class) gets this wrong because it compares directory names in
    isolation, where 'cross' < 'cross-legged...'.

    Fix: collect every .npy path and sort them all together as full strings,
    exactly mirroring the manifest construction.
    """
    all_paths = []
    for cls in os.listdir(root):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.endswith(".npy"):
                all_paths.append(os.path.join(cls_dir, fname))
    yield from sorted(all_paths)  # single sort on full paths — matches manifest


def _enforce_frames(x: np.ndarray, target_t: int) -> np.ndarray:
    if x.ndim != 5:
        raise ValueError(
            f"Expected a 5-D tensor (N, C, T, H, W) but got shape {x.shape}. "
            "Check that your .npy files were saved with the correct format."
        )
    current_t = x.shape[2]
    if current_t < target_t:
        # Pad by repeating the last frame
        x = np.pad(x, ((0,0), (0,0), (0, target_t - current_t), (0,0), (0,0)), mode='edge')
    elif current_t > target_t:
        x = x[:, :, :target_t, :, :]
    return x

def to_channel_last(x: np.ndarray) -> np.ndarray:
    """Transpose (N, C, T, H, W) → (N, T, H, W, C) for channel-last DLC models."""
    return np.transpose(x, (0, 2, 3, 4, 1))  # NCTHW → NTHWC


def load_video_tensors(root: str) -> list[np.ndarray]:
    tensors: list[np.ndarray] = []
    for path in _iter_npy_paths(root):
        x = np.load(path)  # (1, 3, T, H, W) NCTHW
        x = _enforce_frames(x, T)
        if IS_DLC_CHANNEL_LAST:
            x = to_channel_last(x)  # → (1, T, H, W, 3) NTHWC
        tensors.append(x.astype(np.float32))
    return tensors


def load_single_tensor(root: str, single_path: str, index: int) -> np.ndarray:
    if single_path:
        if not os.path.exists(single_path):
            raise FileNotFoundError(f"Single tensor path '{single_path}' not found.")
        x = np.load(single_path)
        x = _enforce_frames(x, T)
        if IS_DLC_CHANNEL_LAST:
            x = to_channel_last(x)
        return x.astype(np.float32)

    paths = list(_iter_npy_paths(root))
    if not paths:
        raise FileNotFoundError(f"No .npy tensors found under '{root}'.")
    if index < 0 or index >= len(paths):
        raise IndexError(f"SINGLE_TENSOR_INDEX out of range (0..{len(paths)-1}).")
    x = np.load(paths[index])
    x = _enforce_frames(x, T)
    if IS_DLC_CHANNEL_LAST:
        x = to_channel_last(x)
    return x.astype(np.float32)


if USE_SINGLE_TENSOR:
    video_tensors = [load_single_tensor(data_path, SINGLE_TENSOR_PATH, SINGLE_TENSOR_INDEX)]
else:
    video_tensors = load_video_tensors(data_path)

print("Loaded", len(video_tensors), "samples")
device = qai_hub.Device(DEVICE_NAME)

# -----------------------------------------------------------------------
# Model loading: DLC (pre-compiled) OR ONNX (compile on QAI Hub)
# -----------------------------------------------------------------------
if DLC_PATH:
    # --- Path A: Upload a pre-compiled DLC directly, skip the compile step ---
    print(f"Using pre-compiled DLC: {DLC_PATH}")
    target_model = qai_hub.upload_model(DLC_PATH)
    print(f"Uploaded model: {target_model}")
else:
    # --- Path B: Load the ONNX model and compile it on QAI Hub ---
    VIDEO_ONNX_PATH = os.path.join(ONNX_DIR, VIDEO_ONNX_NAME)

    print(f"Loading ONNX video model from {VIDEO_ONNX_PATH}...")
    onnx_video_model = onnx.load(VIDEO_ONNX_PATH)

    try:
        onnx.checker.check_model(onnx_video_model)
        print("Video ONNX model is valid ✅")
    except onnx.checker.ValidationError as e:
        print("Video ONNX model validation failed ❌")
        print(e)
        sys.exit(1)

    input_specs = {
        "video": ((BATCH, C, T, H, W), "float32")
    }

    compile_job_id = compile_model(
        model=onnx_video_model,
        device=device,
        input_specs=input_specs,
    )
    target_model = qai_hub.get_job(compile_job_id).get_target_model()

# -----------------------------------------------------------------------
# Dataset upload and inference (chunked for 2GB flatbuffer limit)
# -----------------------------------------------------------------------

CHUNK_SIZE = 538
all_inference_jobs = []

print(f"Submitting dataset in chunks of {CHUNK_SIZE} to avoid size limits...")
for i in range(0, len(video_tensors), CHUNK_SIZE):
    chunk = video_tensors[i : i + CHUNK_SIZE]
    dataset = qai_hub.upload_dataset(
        {"video": chunk},
        name=f"{DATASET_NAME}_part_{i//CHUNK_SIZE + 1}",
    )
    inference_id = inference_job(
        model=target_model,
        device=device,
        dataset=dataset,
    )
    print(f"Chunk {i//CHUNK_SIZE + 1} Inference job ID: {inference_id}")
    all_inference_jobs.append(qai_hub.get_job(inference_id))

print("\nWaiting for all inference jobs and collecting results...")
combined_logits = []

for job in all_inference_jobs:
    job.wait()
    output_data = job.download_output_data()
    key = 'class_probs' if 'class_probs' in output_data else list(output_data.keys())[0]
    combined_logits.extend(output_data[key])

print(f"Successfully collected {len(combined_logits)} inference results.")
print(f"Writing combined output to {OUTPUT_H5} for evaluate.py...")

with h5py.File(OUTPUT_H5, 'w') as f:
    grp = f.create_group("data/0")
    for i, arr in enumerate(combined_logits):
        arr = np.array(arr)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]   
        grp.create_dataset(f'batch_{i}', data=arr)

print(f"Done! Run: python evaluate.py --h5 {OUTPUT_H5}")

