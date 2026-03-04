import argparse
import gc
import os
import sys

import h5py
import numpy as np
import onnx
import qai_hub
from compile_and_profile import compile_model

# =============================================================================
# Utility functions
# =============================================================================


def inference_job(model, device, dataset):
    job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=dataset,
        options="",
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
        x = np.pad(x, ((0, 0), (0, 0), (0, target_t - current_t), (0, 0), (0, 0)), mode="edge")
    elif current_t > target_t:
        x = x[:, :, :target_t, :, :]
    return x


def to_channel_last(x: np.ndarray) -> np.ndarray:
    """Transpose (N, C, T, H, W) → (N, T, H, W, C) for channel-last DLC models."""
    return np.transpose(x, (0, 2, 3, 4, 1))  # NCTHW → NTHWC


def _load_tensor(path: str, target_t: int, is_channel_last: bool) -> np.ndarray:
    """Load a single .npy tensor, enforce frames, and optionally transpose."""
    x = np.load(path)
    x = _enforce_frames(x, target_t)
    if is_channel_last:
        x = to_channel_last(x)
    return x.astype(np.float32)


def load_video_tensors(root: str, target_t: int, is_channel_last: bool) -> list[np.ndarray]:
    tensors: list[np.ndarray] = []
    for path in _iter_npy_paths(root):
        tensors.append(_load_tensor(path, target_t, is_channel_last))
    return tensors


def load_single_tensor(root: str, single_path: str, index: int, target_t: int, is_channel_last: bool) -> np.ndarray:
    if single_path:
        if not os.path.exists(single_path):
            raise FileNotFoundError(f"Single tensor path '{single_path}' not found.")
        return _load_tensor(single_path, target_t, is_channel_last)

    paths = list(_iter_npy_paths(root))
    if not paths:
        raise FileNotFoundError(f"No .npy tensors found under '{root}'.")
    if index < 0 or index >= len(paths):
        raise IndexError(f"SINGLE_TENSOR_INDEX out of range (0..{len(paths) - 1}).")
    return _load_tensor(paths[index], target_t, is_channel_last)


# =============================================================================
# Main
# =============================================================================


def _validate_config(
    dlc_path: str,
    data_path: str,
    onnx_dir: str,
    use_single_tensor: bool,
    single_tensor_path: str,
) -> None:
    """Validate configuration, fail fast before any expensive work."""
    errors = []

    if not data_path:
        errors.append("'data_path' is not set. Point it to the directory of preprocessed .npy tensors.")
    elif not os.path.isdir(data_path):
        errors.append(f"'data_path' directory not found: '{data_path}'")

    if dlc_path:
        if not os.path.exists(dlc_path):
            errors.append(f"DLC_PATH file not found: '{dlc_path}'")
    else:
        if not onnx_dir:
            errors.append(
                "'onnx_dir' is not set and 'dlc_path' is empty. "
                "Either set --dlc-path (pre-compiled model) or set --onnx-dir (ONNX compile path)."
            )
        else:
            onnx_path = os.path.join(onnx_dir, "model.onnx")
            if not os.path.exists(onnx_path):
                errors.append(
                    f"ONNX model not found: '{onnx_path}'. Set --onnx-dir to the correct directory or run export first."
                )

    if use_single_tensor and not single_tensor_path and not data_path:
        errors.append("--use-single-tensor is set but neither --single-tensor-path nor --data-path is provided.")

    if errors:
        print("Configuration error(s) in run_inference.py — please fix before running:\n")
        for err in errors:
            print(f"  ✗ {err}")
        sys.exit(1)


def _load_model(dlc_path: str, onnx_dir: str, device, num_frames: int):
    """Load model from DLC or compile from ONNX."""
    if dlc_path:
        print(f"Using pre-compiled DLC: {dlc_path}")
        target_model = qai_hub.upload_model(dlc_path)
        print(f"Uploaded model: {target_model}")
        return target_model

    video_onnx_path = os.path.join(onnx_dir, "model.onnx")
    print(f"Loading ONNX video model from {video_onnx_path}...")
    onnx_video_model = onnx.load(video_onnx_path)

    try:
        onnx.checker.check_model(onnx_video_model)
        print("Video ONNX model is valid ✅")
    except onnx.checker.ValidationError as e:
        print("Video ONNX model validation failed ❌")
        print(e)
        sys.exit(1)

    input_specs = {"video": ((1, 3, num_frames, 112, 112), "float32")}

    compile_job_id = compile_model(
        model=onnx_video_model,
        device=device,
        input_specs=input_specs,
    )
    return qai_hub.get_job(compile_job_id).get_target_model()


def main(
    dlc_path: str = "",
    data_path: str = "",
    output_h5: str = "dataset-export.h5",
    device_name: str = "Dragonwing IQ-9075 EVK",
    is_dlc_channel_last: bool = True,
    use_single_tensor: bool = False,
    single_tensor_path: str = "",
    single_tensor_index: int = 0,
    chunk_size: int = 538,
    max_samples: int | None = None,
    onnx_dir: str = "",
    num_frames: int = 16,
) -> None:
    """Run AI Hub inference pipeline.

    Args:
        dlc_path: Path to pre-compiled DLC/bin file. Empty to use ONNX path.
        data_path: Directory of preprocessed .npy tensors.
        output_h5: Output HDF5 file path for evaluate.py.
        device_name: AI Hub device name.
        is_dlc_channel_last: True if DLC expects NTHWC input.
        use_single_tensor: If True, run single-tensor inference for debugging.
        single_tensor_path: Path to a specific .npy tensor (for single-tensor mode).
        single_tensor_index: Index of tensor to use from data_path (for single-tensor mode).
        chunk_size: Number of samples per inference chunk (for 2GB flatbuffer limit).
        max_samples: Max number of samples to process. None for all.
        onnx_dir: Directory containing model.onnx (used when dlc_path is empty).
        num_frames: Number of input frames (T dimension).
    """
    _validate_config(dlc_path, data_path, onnx_dir, use_single_tensor, single_tensor_path)

    device = qai_hub.Device(device_name)
    target_model = _load_model(dlc_path, onnx_dir, device, num_frames)

    all_inference_jobs = []

    if use_single_tensor:
        # --- Single-tensor mode (quick debugging) ---
        tensor = load_single_tensor(data_path, single_tensor_path, single_tensor_index, num_frames, is_dlc_channel_last)
        print("Loaded 1 sample (single-tensor mode)")
        dataset = qai_hub.upload_dataset({"video": [tensor]}, name="dataset_single")
        inf_id = inference_job(model=target_model, device=device, dataset=dataset)
        print(f"Inference job ID: {inf_id}")
        all_inference_jobs.append(qai_hub.get_job(inf_id))
    else:
        # --- Chunk-load mode (OOM prevention) ---
        all_npy_paths = list(_iter_npy_paths(data_path))
        if max_samples is not None:
            all_npy_paths = all_npy_paths[:max_samples]
        print(f"Found {len(all_npy_paths)} samples (chunk-load mode)")

        print(f"Submitting dataset in chunks of {chunk_size} to avoid size limits...")
        for i in range(0, len(all_npy_paths), chunk_size):
            chunk_paths = all_npy_paths[i : i + chunk_size]
            chunk_tensors = []
            for p in chunk_paths:
                chunk_tensors.append(_load_tensor(p, num_frames, is_dlc_channel_last))

            dataset = qai_hub.upload_dataset(
                {"video": chunk_tensors},
                name=f"dataset_part_{i // chunk_size + 1}",
            )
            del chunk_tensors
            gc.collect()

            inf_id = inference_job(model=target_model, device=device, dataset=dataset)
            print(f"Chunk {i // chunk_size + 1} Inference job ID: {inf_id}")
            all_inference_jobs.append(qai_hub.get_job(inf_id))

    # --- Collect results ---
    print("\nWaiting for all inference jobs and collecting results...")
    combined_logits = []

    for job in all_inference_jobs:
        job.wait()
        output_data = job.download_output_data()
        key = "class_probs" if "class_probs" in output_data else list(output_data.keys())[0]
        combined_logits.extend(output_data[key])

    print(f"Successfully collected {len(combined_logits)} inference results.")
    print(f"Writing combined output to {output_h5} for evaluate.py...")

    with h5py.File(output_h5, "w") as f:
        grp = f.create_group("data/0")
        for i, arr in enumerate(combined_logits):
            arr = np.array(arr)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            grp.create_dataset(f"batch_{i}", data=arr)

    print(f"Done! Run: python evaluate.py --h5 {output_h5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Hub inference pipeline")
    parser.add_argument("--dlc-path", type=str, default="", help="Path to pre-compiled DLC/bin file")
    parser.add_argument("--data-path", type=str, default="", help="Directory of preprocessed .npy tensors")
    parser.add_argument("--output-h5", type=str, default="dataset-export.h5", help="Output HDF5 file path")
    parser.add_argument("--device-name", type=str, default="Dragonwing IQ-9075 EVK", help="AI Hub device name")
    parser.add_argument(
        "--channel-last",
        dest="is_dlc_channel_last",
        action="store_true",
        default=True,
        help="DLC expects NTHWC input (default: True)",
    )
    parser.add_argument(
        "--no-channel-last", dest="is_dlc_channel_last", action="store_false", help="DLC expects NCTHW input"
    )
    parser.add_argument(
        "--use-single-tensor", action="store_true", default=False, help="Single-tensor mode for debugging"
    )
    parser.add_argument("--single-tensor-path", type=str, default="", help="Path to specific .npy tensor")
    parser.add_argument("--single-tensor-index", type=int, default=0, help="Index of tensor to use from data_path")
    parser.add_argument("--chunk-size", type=int, default=538, help="Samples per inference chunk")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--onnx-dir", type=str, default="", help="Directory containing model.onnx")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of input frames")
    args = parser.parse_args()
    main(**vars(args))
