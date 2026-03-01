import json
import os
from typing import Iterable
import numpy as np
import torch
from video_processing import process_video


# =============================================================================
# USER CONFIGURATION — update these values before running
# =============================================================================

# Root directory of the raw video dataset.
# Expected structure: DATA_ROOT/<class_name>/<video>.mp4
# The class folder name is used as the label in the manifest.
DATA_ROOT = ""

# Directory where preprocessed .npy tensors and the manifest will be saved.
OUT_ROOT = ""

# Video file extensions to include when scanning DATA_ROOT.
VIDEO_EXTS = {".mp4"}

# Number of frames to sample per clip. Must match the model's expected input.
CLIP_LEN = 16

# Target frame rate used when sampling frames from each video.
FRAME_RATE = 4

# =============================================================================


def list_videos(root: str) -> list[str]:
    videos: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            _, ext = os.path.splitext(name)
            if ext.lower() in VIDEO_EXTS:
                videos.append(os.path.join(dirpath, name))
    return sorted(videos)


def iter_with_label(videos: Iterable[str], root: str) -> Iterable[tuple[str, str]]:
    for path in videos:
        rel = os.path.relpath(path, root)
        parts = rel.split(os.sep)
        label = parts[0] if len(parts) > 1 else "unknown"
        yield path, label


def save_tensor_npy(tensor: torch.Tensor, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, tensor.detach().cpu().numpy())


def main() -> None:
    if not DATA_ROOT:
        raise ValueError("'DATA_ROOT' is not set. Update it at the top of this script.")
    if not OUT_ROOT:
        raise ValueError("'OUT_ROOT' is not set. Update it at the top of this script.")

    videos = list_videos(DATA_ROOT)
    if not videos:
        raise FileNotFoundError(
            f"No {', '.join(VIDEO_EXTS)} files found under '{DATA_ROOT}'. "
            "Check that DATA_ROOT points to the correct directory."
        )

    manifest_path = os.path.join(OUT_ROOT, "manifest.jsonl")
    os.makedirs(OUT_ROOT, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for video_path, label in iter_with_label(videos, DATA_ROOT):
            rel = os.path.relpath(video_path, DATA_ROOT)
            rel_no_ext = os.path.splitext(rel)[0]
            out_path = os.path.join(OUT_ROOT, f"{rel_no_ext}.npy")

            clip = process_video(
                video_path=video_path,
                batch_size=1,
                clip_len=CLIP_LEN,
                frame_rate=FRAME_RATE,
                clip_strategy="uniform",
                device=torch.device("cpu"),
                output_dtype=torch.float32,  # Save as float32 
            )

            # clip: (1, 3, T, 112, 112)
            save_tensor_npy(clip, out_path)

            record = {
                "video_path": video_path,
                "label": label,
                "tensor_path": out_path,
                "shape": list(clip.shape),
                "dtype": str(clip.dtype),
            }
            manifest.write(json.dumps(record) + "\n")

    print(f"Wrote tensors to {OUT_ROOT}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
