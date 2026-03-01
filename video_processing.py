import torch
import torchvision.transforms as transforms
from torchvision.datasets.video_utils import VideoClips


def process_video(
    video_path: str,
    batch_size: int = 1,
    clip_len: int = 16,
    frame_rate: int = 4,
    clip_strategy: str = "uniform",
    device: torch.device | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load a video and return a batch of preprocessed clips for r2plus1d_18.

    Output shape: (B, 3, T, 112, 112), dtype `output_dtype`.

    Preprocessing pipeline:
      1. Decode frames at `frame_rate` fps.
      2. Convert uint8 pixel values to float32 in [0, 1].
      3. Resize the shorter side to 128×171, then centre-crop to 112×112.

    Note: mean/std normalisation (mean=(0.43216, 0.394666, 0.37645),
    std=(0.22803, 0.22145, 0.216989)) is intentionally NOT applied here —
    it is baked into the exported ONNX graph (inside NormalizedVideoModel).
    Applying it twice would produce incorrect results.

    Args:
        video_path:     Path to the input video file.
        batch_size:     Number of clips to sample from the video.
        clip_len:       Number of frames per clip. Must match the model input.
        frame_rate:     Target frames per second used when decoding.
        clip_strategy:  How to pick clips from the video:
                          "uniform" — evenly spaced clips (recommended),
                          "first"   — always use the first clip.
        device:         Torch device to move the output tensor to.
        output_dtype:   Output tensor dtype (default: float32).

    Returns:
        Tensor of shape (batch_size, 3, clip_len, 112, 112).
    """
    if clip_strategy not in ("uniform", "first"):
        raise ValueError(
            f"Unknown clip_strategy '{clip_strategy}'. "
            "Choose 'uniform' or 'first'."
        )

    device = device or torch.device("cpu")

    video_clips = VideoClips(
        [video_path],
        clip_length_in_frames=clip_len,
        frames_between_clips=1,
        frame_rate=frame_rate,
        output_format="TCHW",
    )

    num_clips = video_clips.num_clips()
    if num_clips < 1:
        raise ValueError(
            f"Not enough frames in '{video_path}' to form a clip of length "
            f"{clip_len} at {frame_rate} fps. "
            "Try reducing clip_len or frame_rate."
        )

    spatial = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),  # uint8 → [0, 1] float
            transforms.Resize((128, 171), antialias=False),
            transforms.CenterCrop((112, 112)),
        ]
    )

    if clip_strategy == "uniform":
        clip_indices = (
            torch.linspace(0, num_clips - 1, steps=batch_size)
            .floor()
            .to(torch.int64)
            .tolist()
        )
    else:  # "first"
        clip_indices = [0] * batch_size

    clips = []
    for idx in clip_indices:
        clip, _, _, _ = video_clips.get_clip(idx)
        # clip: (T, C, H, W) → apply spatial transforms → permute to (C, T, H, W)
        clip = spatial(clip).permute(1, 0, 2, 3)
        clips.append(clip)

    # Stack into (B, C, T, H, W) and move to target device/dtype
    return torch.stack(clips, dim=0).to(device=device, dtype=output_dtype)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_processing.py <path_to_video.mp4>")
        sys.exit(1)

    video_path = sys.argv[1]
    batch = process_video(video_path, batch_size=1, clip_len=16, frame_rate=4)
    print(f"Output shape : {tuple(batch.shape)}")   # expected: (1, 3, 16, 112, 112)
    print(f"dtype        : {batch.dtype}")
    print(f"value range  : [{batch.min():.4f}, {batch.max():.4f}]")
