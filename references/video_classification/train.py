from __future__ import annotations

import datetime
import io
import json
import logging
import math
import os
import pickle
import random
import sys
import time
import warnings
from pathlib import Path

# src/models.py をインポートするためにリポルートの src/ を sys.path に追加
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.datasets.video_utils
import torchvision.transforms as T
import utils
from datasets import KineticsWithVideoId
from PIL import Image
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers import (
    DistributedSampler,
    RandomClipSampler,
    UniformClipSampler,
)

# WebDataset (オプション)
try:
    import webdataset as wds

    HAS_WEBDATASET = True
except ImportError:
    HAS_WEBDATASET = False


# =============================================================================
# WebDataset 用ヘルパークラス
# =============================================================================


class WebDatasetClipSampler:
    """
    WebDatasetのフレームリストからランダムクリップをサンプリング。
    mp4直接読み込み (VideoClips) と同等の動作を再現。

    VideoClips._resample_video_idx と同じロジックを使用:
    - 動画のFPSとターゲットframe_rateを考慮してリサンプリング
    - 短い動画は動的にframe_rateを調整
    """

    def __init__(
        self,
        frames_per_clip: int = 8,
        target_frame_rate: int = 4,
        is_train: bool = True,
    ):
        self.frames_per_clip = frames_per_clip
        self.target_frame_rate = target_frame_rate
        self.is_train = is_train

    def _resample_video_idx(self, num_output_frames: int, original_fps: float, new_fps: float) -> list[int]:
        """
        VideoClips._resample_video_idx と同等のロジック。

        Args:
            num_output_frames: リサンプリング後の出力フレーム数
            original_fps: 元の動画のFPS
            new_fps: ターゲットFPS

        Returns:
            元の動画フレームへのインデックスリスト (長さ = num_output_frames)
        """
        import math

        step = original_fps / new_fps
        if step == int(step):
            # 整数ステップの場合
            step = int(step)
            return [i * step for i in range(num_output_frames)]
        else:
            # 非整数ステップ: torch.arange(num_frames) * step と同等
            idxs = [int(math.floor(i * step)) for i in range(num_output_frames)]
            return idxs

    def __call__(self, frames: list[bytes], fps: float) -> list[bytes]:
        """
        Args:
            frames: 全フレームのJPEGバイト列リスト
            fps: 動画の元FPS

        Returns:
            サンプリングされたフレームのリスト (frames_per_clip個)
        """
        import math

        total_frames = len(frames)
        target_fps = self.target_frame_rate

        # VideoClips.compute_clips_for_video と同じロジック
        # リサンプリング後の総フレーム数を計算
        resampled_total = total_frames * target_fps / fps

        # 短い動画の場合、frame_rateを動的に調整 (video_utils.py:218-223)
        if resampled_total < self.frames_per_clip:
            video_duration = total_frames / fps
            resampled_total = self.frames_per_clip
            target_fps = math.ceil(self.frames_per_clip / video_duration)

        # リサンプリングインデックスを計算
        resampled_indices = self._resample_video_idx(int(math.floor(resampled_total)), fps, target_fps)

        # クリップ可能な範囲を計算
        num_resampled = len(resampled_indices)
        if num_resampled < self.frames_per_clip:
            # それでも足りない場合はインデックスを繰り返し
            repeat_count = (self.frames_per_clip // num_resampled) + 1
            resampled_indices = (resampled_indices * repeat_count)[: self.frames_per_clip]
            num_resampled = len(resampled_indices)

        max_start = num_resampled - self.frames_per_clip

        if self.is_train:
            start = random.randint(0, max(0, max_start))
        else:
            start = max(0, max_start) // 2  # 中央からサンプリング

        # クリップのインデックスを取得
        clip_indices = resampled_indices[start : start + self.frames_per_clip]

        # フレーム範囲を超えないようにクリップ
        clip_indices = [min(idx, total_frames - 1) for idx in clip_indices]

        return [frames[idx] for idx in clip_indices]


class WebDatasetFrameTransform:
    """
    WebDataset用フレーム変換。presets.VideoClassificationPreset と同等の順序。

    重要: RandomCrop/RandomHorizontalFlip はクリップ内の全フレームに同じ変換を適用する必要がある。
    フレームごとに異なるクロップ/フリップをすると時間的一貫性が壊れてモデルが動きを学習できない。

    そのため per-frame 変換 (ToTensor, Normalize) と clip-level 変換 (Crop, Flip) を分離。
    ColorJitter は clip 内の全フレームに同一パラメータで適用する (temporal consistency 維持)。
    """

    def __init__(
        self,
        crop_size: tuple[int, int] = (112, 112),
        is_train: bool = True,
        color_jitter: tuple[float, ...] | None = None,
        random_resized_crop_scale: tuple[float, float] | None = None,
    ):
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]

        # per-frame: 各フレームに独立適用 (ランダム性なし)
        # Normalize は ColorJitter の後に適用するため分離
        self.per_frame = T.ToTensor()
        self.normalize = T.Normalize(mean=mean, std=std)

        # ColorJitter: clip内の全フレームに同一パラメータで適用
        # 重要: [0,1] 範囲のテンソルに適用する必要がある (Normalize 前)
        self.color_jitter = None
        if is_train and color_jitter is not None:
            self.color_jitter = T.ColorJitter(*color_jitter)

        # clip-level: スタック後の [T,C,H,W] テンソルに適用 (全フレーム同じ変換)
        if is_train:
            crop_transforms: list = [T.RandomHorizontalFlip()]
            if random_resized_crop_scale is not None:
                crop_transforms.append(
                    T.RandomResizedCrop(
                        crop_size,
                        scale=random_resized_crop_scale,
                        ratio=(0.75, 1.3333),
                        antialias=False,
                    )
                )
            else:
                crop_transforms.append(T.RandomCrop(crop_size))
            self.clip_transform = T.Compose(crop_transforms)
        else:
            self.clip_transform = T.CenterCrop(crop_size)

    def transform_frame(self, img: Image.Image) -> torch.Tensor:
        """1フレームを変換 (ランダム性なし)"""
        return self.per_frame(img)

    def transform_clip(self, video: torch.Tensor) -> torch.Tensor:
        """クリップ全体に変換.

        適用順序: ColorJitter ([0,1]範囲) → Normalize → Crop/Flip
        ColorJitter は全フレーム同一パラメータで temporal consistency を維持。
        """
        # 1. ColorJitter (入力は [0,1] 範囲のテンソル)
        if self.color_jitter is not None:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = T.ColorJitter.get_params(
                self.color_jitter.brightness,
                self.color_jitter.contrast,
                self.color_jitter.saturation,
                self.color_jitter.hue,
            )
            from torchvision.transforms import functional as F

            for i in range(video.shape[0]):
                frame = video[i]  # [C, H, W], [0,1] 範囲
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        frame = F.adjust_brightness(frame, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        frame = F.adjust_contrast(frame, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        frame = F.adjust_saturation(frame, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        frame = F.adjust_hue(frame, hue_factor)
                video[i] = frame

        # 2. Normalize (ColorJitter の後に適用)
        for i in range(video.shape[0]):
            video[i] = self.normalize(video[i])

        # 3. Crop/Flip
        return self.clip_transform(video)


def make_webdataset_transform(
    clip_sampler: WebDatasetClipSampler,
    frame_transform: WebDatasetFrameTransform,
    class_to_idx: dict[str, int],
):
    """
    WebDatasetのサンプル変換関数を作成。
    mp4版と同じ出力形式: (video, audio, label, video_idx)

    新フォーマット対応: frames.pkl に {"frames": [...], "fps": float} を格納
    """

    def transform(sample: dict) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        # フレームとFPSをデコード
        frames_pkl = sample["frames.pkl"]
        frame_data = pickle.loads(frames_pkl)

        # 新旧フォーマット対応
        if isinstance(frame_data, dict):
            frames: list[bytes] = frame_data["frames"]
            fps: float = frame_data["fps"]
        else:
            # 旧フォーマット (リストのみ) の場合はデフォルトFPS
            frames = frame_data
            fps = 30.0

        # FPSを考慮したクリップサンプリング
        sampled_frames = clip_sampler(frames, fps)

        # 各フレームを変換 (per-frame: ToTensor + Normalize のみ)
        tensors = []
        for frame_bytes in sampled_frames:
            img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            tensors.append(frame_transform.transform_frame(img))

        # clip-level 変換 (Crop/Flip を全フレーム同一座標で適用)
        video = torch.stack(tensors, dim=0)  # [T, C, H, W]
        video = frame_transform.transform_clip(video)  # [T, C, H, W] (cropped)
        video = video.permute(1, 0, 2, 3)  # [C, T, H, W]

        # ラベル
        label_str = sample["cls.txt"].decode("utf-8")
        label = class_to_idx[label_str]

        # 空のaudioテンソルとダミーのvideo_idx
        audio = torch.empty(0)
        video_idx = 0

        return video, audio, label, video_idx

    return transform


def create_webdataset_loader(
    shards: str,
    class_to_idx: dict[str, int],
    batch_size: int,
    num_workers: int,
    is_train: bool,
    crop_size: tuple[int, int],
    frames_per_clip: int,
    frame_rate: int,
    distributed: bool = False,
    epoch_length: int | None = None,
    color_jitter: tuple[float, ...] | None = None,
    random_resized_crop_scale: tuple[float, float] | None = None,
) -> torch.utils.data.DataLoader:
    """
    WebDataset DataLoaderを作成。
    """
    if not HAS_WEBDATASET:
        raise ImportError("webdataset is required. Install with: pip install webdataset")

    clip_sampler = WebDatasetClipSampler(
        frames_per_clip=frames_per_clip,
        target_frame_rate=frame_rate,
        is_train=is_train,
    )
    frame_transform = WebDatasetFrameTransform(
        crop_size=crop_size,
        is_train=is_train,
        color_jitter=color_jitter,
        random_resized_crop_scale=random_resized_crop_scale,
    )
    sample_transform = make_webdataset_transform(clip_sampler, frame_transform, class_to_idx)

    # 分散学習の場合、nodesplitterでシャードをランク間で分割
    nodesplitter = wds.split_by_node if distributed else wds.shardlists.single_node_only

    if is_train:
        # empty_check=False: シャード数 < workers の場合に一部workerが空でもエラーにしない
        # .repeat(): epoch_lengthで切るため無限にサンプリング
        # .with_epoch(): 全rankが同じイテレーション数で終了 (NCCLデッドロック回避)
        dataset = (
            wds.WebDataset(shards, shardshuffle=True, nodesplitter=nodesplitter, empty_check=False)
            .shuffle(1000)
            .map(sample_transform)
            .repeat()
            .with_epoch(epoch_length if epoch_length else 10000)
        )
    else:
        # Validation: split_by_nodeで各rankが別シャードを処理
        # シャード数 >= GPU数が前提 (valシャード不足の場合はvalデータを事前に作成すること)
        # .repeat() + .with_epoch() で全rankが同じイテレーション数を保証
        dataset = (
            wds.WebDataset(shards, shardshuffle=False, nodesplitter=nodesplitter, empty_check=False)
            .map(sample_transform)
            .repeat()
            .with_epoch(epoch_length if epoch_length else 1000)
        )

    # 分散学習のvalidationではworker=0を強制 (worker間シャード分割の問題回避)
    actual_workers = num_workers if is_train else 0
    persistent = is_train and actual_workers > 0

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=actual_workers,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=collate_fn,
        persistent_workers=persistent,
    )

    return loader


# =============================================================================
# マルチクリップ評価用 (WebDataset)
# =============================================================================


def generate_all_clips(
    frames: list[bytes],
    fps: float,
    frames_per_clip: int,
    target_frame_rate: int,
) -> list[list[bytes]]:
    """1動画から全クリップを生成 (VideoClips の step_between_clips=1 と同等).

    Args:
        frames: 全フレームの JPEG バイト列リスト.
        fps: 元の FPS.
        frames_per_clip: クリップあたりフレーム数.
        target_frame_rate: ターゲット FPS.

    Returns:
        クリップのリスト。各クリップはフレームバイト列のリスト.
    """
    total_frames = len(frames)
    target_fps = target_frame_rate

    # VideoClips.compute_clips_for_video と同じロジック
    resampled_total = total_frames * target_fps / fps

    if resampled_total < frames_per_clip:
        video_duration = total_frames / fps
        resampled_total = frames_per_clip
        target_fps = math.ceil(frames_per_clip / video_duration)

    num_resampled = int(math.floor(resampled_total))

    # リサンプリングインデックス (_resample_video_idx と同等)
    step = fps / target_fps
    if step == int(step):
        step_int = int(step)
        resampled_indices = [i * step_int for i in range(num_resampled)]
    else:
        resampled_indices = [int(math.floor(i * step)) for i in range(num_resampled)]
    resampled_indices = [min(idx, total_frames - 1) for idx in resampled_indices]

    clips: list[list[bytes]] = []
    if num_resampled < frames_per_clip:
        # フレーム不足: 繰り返しで1クリップのみ
        repeat_count = (frames_per_clip // num_resampled) + 1
        padded = (resampled_indices * repeat_count)[:frames_per_clip]
        clips.append([frames[i] for i in padded])
    else:
        # step_between_clips=1: 全開始位置からクリップ生成
        for start in range(num_resampled - frames_per_clip + 1):
            clip_indices = resampled_indices[start : start + frames_per_clip]
            clips.append([frames[i] for i in clip_indices])

    return clips


def _multiclip_collate_fn(batch: list) -> tuple:
    """マルチクリップ用 collate (video_key を文字列リストとして保持)."""
    videos = torch.stack([b[0] for b in batch])
    audios = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch])
    video_keys = [b[3] for b in batch]
    return videos, audios, labels, video_keys


def create_webdataset_multiclip_loader(
    shards: list[str],
    class_to_idx: dict[str, int],
    batch_size: int,
    num_workers: int,
    crop_size: tuple[int, int],
    frames_per_clip: int,
    frame_rate: int,
    distributed: bool = False,
) -> torch.utils.data.DataLoader:
    """マルチクリップ評価用 WebDataset DataLoader.

    全動画から全クリップを走査し video_key で動画を識別する。
    MP4 版の SequentialSampler + evaluate_pytorch_mp4 と同等の動作。
    """
    if not HAS_WEBDATASET:
        raise ImportError("webdataset is required. Install with: pip install webdataset")

    frame_transform = WebDatasetFrameTransform(crop_size=crop_size, is_train=False)

    def multiclip_pipeline(samples):
        """1動画 → N クリップに展開する compose パイプライン."""
        for sample in samples:
            frame_data = pickle.loads(sample["frames.pkl"])
            if isinstance(frame_data, dict):
                frames: list[bytes] = frame_data["frames"]
                fps: float = frame_data["fps"]
            else:
                frames = frame_data
                fps = 30.0

            label_str = sample["cls.txt"].decode("utf-8")
            label = class_to_idx[label_str]
            video_key = sample["__key__"]

            for clip_frames in generate_all_clips(frames, fps, frames_per_clip, frame_rate):
                tensors = []
                for fb in clip_frames:
                    img = Image.open(io.BytesIO(fb)).convert("RGB")
                    tensors.append(frame_transform.transform_frame(img))

                video = torch.stack(tensors, dim=0)  # [T, C, H, W]
                video = frame_transform.transform_clip(video)  # [T, C, H, W]
                video = video.permute(1, 0, 2, 3)  # [C, T, H, W]

                audio = torch.empty(0)
                yield video, audio, label, video_key

    # 分散学習: split_by_node でシャードを各ランクに分割して並列評価
    nodesplitter = wds.split_by_node if distributed else wds.shardlists.single_node_only
    dataset = wds.WebDataset(
        shards, shardshuffle=False, empty_check=False,
        nodesplitter=nodesplitter,
    ).compose(multiclip_pipeline)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_multiclip_collate_fn,
    )


def train_one_epoch(
    model,
    criterion,
    optimizer,
    lr_scheduler,
    data_loader,
    device,
    epoch,
    print_freq,
    scaler=None,
    total_iters=None,
    mixup_cutmix=None,
    ema_model=None,
):
    """
    Args:
        total_iters: 総イテレーション数 (WebDatasetなどlen()未サポートの場合に指定)
        mixup_cutmix: MixUp/CutMix変換 (Noneの場合は適用しない)
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}"))

    header = f"Epoch: [{epoch}]"
    iteration = 0
    for video, _, target, _ in metric_logger.log_every(data_loader, print_freq, header, total=total_iters):
        start_time = time.time()
        video, target = video.to(device), target.to(device)
        if mixup_cutmix is not None:
            # MixUp/CutMix は 4D (B,C,H,W) を期待するため、
            # 5D (B,C,T,H,W) を一時的に (B,C*T,H,W) に reshape して適用する。
            # CutMix の空間マスクは全フレームで共通になり意味的に正しい。
            B, C, T, H, W = video.shape
            video_4d = video.reshape(B, C * T, H, W)
            video_4d, target = mixup_cutmix(video_4d, target)
            video = video_4d.reshape(B, C, T, H, W)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(video)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # MixUp/CutMix 適用時は target が soft label (B, num_classes) になるため argmax で戻す
        acc_target = target.argmax(dim=1) if target.ndim == 2 else target
        acc1, acc5 = utils.accuracy(output, acc_target, topk=(1, 5))
        batch_size = video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(batch_size / (time.time() - start_time))
        if ema_model is not None:
            ema_model.update_parameters(model)
        lr_scheduler.step()

        # WebDataset (.repeat()) は無限なので、total_itersで明示的に終了
        iteration += 1
        if total_iters is not None and iteration >= total_iters:
            break


def evaluate_webdataset(model, criterion, data_loader, device, total_iters=None):
    """
    WebDataset用の評価関数。クリップレベルの精度のみ計算。
    (WebDatasetはvideo_idxをサポートしないため、動画レベル集計は行わない)

    Args:
        total_iters: 総イテレーション数 (WebDatasetなどlen()未サポートの場合に指定)
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    with torch.inference_mode():
        iteration = 0
        for video, _, target, _ in metric_logger.log_every(data_loader, 100, header, total=total_iters):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(video)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

            # WebDataset (.repeat()) は無限なので、total_itersで明示的に終了
            iteration += 1
            if total_iters is not None and iteration >= total_iters:
                break

    metric_logger.synchronize_between_processes()

    # データが空の場合 (分散学習で一部rankにデータが届かない場合) のガード
    if "acc1" not in metric_logger.meters:
        print(" * Clip Acc@1 N/A Clip Acc@5 N/A (no data on this rank)")
        return 0.0

    print(f" * Clip Acc@1 {metric_logger.acc1.global_avg:.3f} Clip Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def evaluate_webdataset_multiclip(model, data_loader, device, distributed=False):
    """マルチクリップ WebDataset 評価 (clip + video 精度).

    全クリップを走査し、動画ごとに softmax 平均で集計する。
    MP4 版の evaluate() と同等の評価方法。

    分散学習: 各ランクが担当シャードを評価し、結果を gather して集計。

    Returns:
        video_acc1 (float): 動画レベル Top-1 精度.
    """
    from collections import defaultdict

    model.eval()
    all_outputs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_video_keys: list[str] = []

    t0 = time.time()
    rank = torch.distributed.get_rank() if distributed else 0
    print(f"[multiclip_val] rank {rank} start")
    with torch.inference_mode():
        for i, batch in enumerate(data_loader):
            video, _, target, video_keys = batch
            video = video.to(device)
            output = model(video)
            all_outputs.append(output.cpu())
            all_targets.append(target)
            all_video_keys.extend(video_keys)
            if (i + 1) % 100 == 0 and rank == 0:
                print(f"[multiclip_val] {i + 1} batches, {len(all_video_keys)} clips")

    local_outputs = torch.cat(all_outputs) if all_outputs else torch.empty(0)
    local_targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)

    # 分散: 各ランクの結果を rank 0 に gather
    if distributed:
        import pickle as _pickle

        # outputs, targets, video_keys をシリアライズして gather
        local_data = _pickle.dumps((local_outputs, local_targets, all_video_keys))
        local_bytes = torch.ByteTensor(list(local_data)).to(device)
        local_size = torch.tensor([len(local_data)], device=device)

        # 全ランクのサイズを収集
        world_size = torch.distributed.get_world_size()
        all_sizes = [torch.tensor([0], device=device) for _ in range(world_size)]
        torch.distributed.all_gather(all_sizes, local_size)

        max_size = max(s.item() for s in all_sizes)
        # パディング
        padded = torch.zeros(max_size, dtype=torch.uint8, device=device)
        padded[: len(local_data)] = local_bytes
        all_padded = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
        torch.distributed.all_gather(all_padded, padded)

        # rank 0 でデシリアライズして統合
        if rank == 0:
            merged_outputs = []
            merged_targets = []
            merged_keys: list[str] = []
            for r in range(world_size):
                sz = all_sizes[r].item()
                data = bytes(all_padded[r][:sz].cpu().tolist())
                r_outputs, r_targets, r_keys = _pickle.loads(data)
                merged_outputs.append(r_outputs)
                merged_targets.append(r_targets)
                merged_keys.extend(r_keys)
            all_outputs_t = torch.cat(merged_outputs)
            all_targets_t = torch.cat(merged_targets)
            all_video_keys = merged_keys
        else:
            all_outputs_t = local_outputs
            all_targets_t = local_targets
    else:
        all_outputs_t = local_outputs
        all_targets_t = local_targets

    total_clips = len(all_video_keys)

    # clip accuracy
    clip_pred = all_outputs_t.argmax(dim=1)
    clip_correct = (clip_pred == all_targets_t).sum().item()
    clip_acc1 = 100.0 * clip_correct / max(total_clips, 1)

    # video accuracy (softmax 平均で集計)
    video_preds: dict[str, list[torch.Tensor]] = defaultdict(list)
    video_labels: dict[str, int] = {}
    for idx, key in enumerate(all_video_keys):
        video_preds[key].append(all_outputs_t[idx])
        video_labels[key] = all_targets_t[idx].item()

    video_correct = 0
    for key in video_preds:
        logits = torch.stack(video_preds[key])
        avg_prob = torch.softmax(logits, dim=1).mean(dim=0)
        if avg_prob.argmax().item() == video_labels[key]:
            video_correct += 1

    num_videos = len(video_preds)
    video_acc1 = 100.0 * video_correct / max(num_videos, 1)

    elapsed = time.time() - t0
    if rank == 0:
        print(
            f"[multiclip_val] Clip Acc@1 {clip_acc1:.3f} | Video Acc@1 {video_acc1:.3f}"
            f" ({total_clips} clips, {num_videos} videos, {elapsed:.1f}s)"
        )
        print(f" * Clip Acc@1 {clip_acc1:.3f} Video Acc@1 {video_acc1:.3f} ({total_clips} clips, {num_videos} videos)")

    # video_acc1 を全ランクに broadcast (checkpoint 保存で全ランクが同じ値を使うため)
    if distributed:
        acc_tensor = torch.tensor([video_acc1], device=device)
        torch.distributed.broadcast(acc_tensor, src=0)
        video_acc1 = acc_tensor.item()

    return video_acc1


def evaluate(model, criterion, data_loader, num_classes, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    # Group and aggregate output of a video
    num_videos = len(data_loader.dataset.samples)
    print(f"Evaluating {num_videos} videos\n")
    # num_classes = len(data_loader.dataset.classes)
    agg_preds = torch.zeros((num_videos, num_classes), dtype=torch.float32, device=device)
    agg_targets = torch.zeros((num_videos), dtype=torch.int32, device=device)
    with torch.inference_mode():
        for video, _, target, video_idx in metric_logger.log_every(data_loader, 100, header):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            # Use softmax to convert output into prediction probability
            preds = torch.softmax(output, dim=1)
            for b in range(video.size(0)):
                idx = video_idx[b].item()
                agg_preds[idx] += preds[b].detach()
                agg_targets[idx] = target[b].detach().item()

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if isinstance(data_loader.sampler, DistributedSampler):
        # Get the len of UniformClipSampler inside DistributedSampler
        num_data_from_sampler = len(data_loader.sampler.dataset)
    else:
        num_data_from_sampler = len(data_loader.sampler)

    if (
        hasattr(data_loader.dataset, "__len__")
        and num_data_from_sampler != num_processed_samples
        and utils.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the sampler has {num_data_from_sampler} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f" * Clip Acc@1 {metric_logger.acc1.global_avg:.3f} Clip Acc@5 {metric_logger.acc5.global_avg:.3f}")
    # Reduce the agg_preds and agg_targets from all gpus (safe no-op on single gpu / cpu)
    if utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()
        torch.distributed.all_reduce(agg_preds, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(agg_targets, op=torch.distributed.ReduceOp.MAX)
    agg_acc1, agg_acc5 = utils.accuracy(agg_preds, agg_targets, topk=(1, 5))
    print(f" * Video Acc@1 {agg_acc1:.3f} Video Acc@5 {agg_acc5:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath, args):
    import hashlib

    value = f"{filepath}-{args.clip_len}-{args.kinetics_version}-{args.frame_rate}"
    h = hashlib.sha1(value.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def collate_fn(batch):
    # remove audio from the batch
    return default_collate(batch)


def setup_logging(output_dir: str, rank: int = 0) -> logging.Logger:
    """
    ログディレクトリを作成し、ファイル・コンソール両方に出力するロガーを設定。
    rank 0 のみファイル出力を行う (分散学習対応)。
    """
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # コンソール出力
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # ファイル出力 (rank 0 のみ)
    if rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"train_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_fmt)
        logger.addHandler(file_handler)

        # メトリクス用 JSON Lines ファイル
        metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.jsonl")
        logger.metrics_file = metrics_file
    else:
        logger.metrics_file = None

    return logger


def log_metrics(logger: logging.Logger, epoch: int, metrics: dict) -> None:
    """メトリクスを JSON Lines 形式でファイルに追記。"""
    if hasattr(logger, "metrics_file") and logger.metrics_file:
        record = {"epoch": epoch, **metrics}
        with open(logger.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    # ロガー設定
    rank = getattr(args, "rank", 0)
    logger = setup_logging(args.output_dir, rank) if args.output_dir else logging.getLogger("train")
    logger.info(f"Args: {args}")

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Data loading code
    print("Loading data")
    val_crop_size = tuple(args.val_crop_size)
    train_crop_size = tuple(args.train_crop_size)

    # =========================================================================
    # データソースの分岐: mp4 or webdataset
    # =========================================================================
    if args.data_source == "webdataset":
        # WebDataset モード
        print(f"Using WebDataset: {args.webdataset_path}")

        # クラス一覧を読み込み
        classes_file = os.path.join(args.webdataset_path, "classes.txt")
        with open(classes_file) as f:
            classes = [line.strip() for line in f if line.strip()]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        num_classes = len(classes)
        print(f"Classes: {num_classes}")

        # シャードパターン (存在するシャードを自動検出)
        import glob

        if not args.test_only:
            train_shard_files = sorted(glob.glob(os.path.join(args.webdataset_path, "train", "shard-*.tar")))
            if not train_shard_files:
                raise FileNotFoundError(f"No train shards found in {args.webdataset_path}/train/")
            train_shards = train_shard_files  # リストで渡す
        val_shard_files = sorted(glob.glob(os.path.join(args.webdataset_path, "val", "shard-*.tar")))
        if not val_shard_files:
            raise FileNotFoundError(
                f"No val shards found in {args.webdataset_path}/val/. "
                "Create val webdataset first with create_webdataset.py --split val"
            )
        val_shards = val_shard_files
        if not args.test_only:
            print(f"Found {len(train_shard_files)} train shards, {len(val_shard_files)} val shards")
        else:
            print(f"Found {len(val_shard_files)} val shards (test-only mode)")

        # val サンプル数を取得 (引数 or meta.txt から自動取得)
        if args.webdataset_val_samples > 0:
            val_samples = args.webdataset_val_samples
        else:
            val_meta_file = os.path.join(args.webdataset_path, "val", "meta.txt")
            if os.path.exists(val_meta_file):
                with open(val_meta_file) as f:
                    for line in f:
                        if line.startswith("num_samples:"):
                            val_samples = int(line.split(":")[1].strip())
                            break
                    else:
                        val_samples = 10000  # fallback
            else:
                val_samples = 10000  # fallback
        print(f"Val samples: {val_samples}")

        # 分散学習で各rankが同じイテレーション数になるようepoch_lengthを設定
        # (NCCLデッドロック回避のため必須)
        if not args.test_only:
            train_epoch_length = args.webdataset_samples // max(1, args.world_size)
        # val 全体を使用 (切り上げで全サンプルをカバー)
        val_epoch_length = math.ceil(val_samples / max(1, args.world_size))

        # augmentation パラメータ
        _color_jitter = None
        cj = getattr(args, "color_jitter", None)
        if cj:
            if len(cj) >= 3:
                _color_jitter = tuple(cj)
            else:
                logger.warning(
                    f"color_jitter requires >= 3 values (brightness, contrast, saturation[, hue]), "
                    f"got {len(cj)}. Ignoring."
                )
        _rrc_scale = None
        rrc = getattr(args, "random_resized_crop_scale", None)
        if rrc:
            if len(rrc) == 2:
                _rrc_scale = tuple(rrc)
            else:
                logger.warning(
                    f"random_resized_crop_scale requires exactly 2 values (min, max), got {len(rrc)}. Ignoring."
                )

        if not args.test_only:
            print("Creating WebDataset data loaders")
            data_loader = create_webdataset_loader(
                shards=train_shards,
                class_to_idx=class_to_idx,
                batch_size=args.batch_size,
                num_workers=args.workers,
                is_train=True,
                crop_size=train_crop_size,
                frames_per_clip=args.clip_len,
                frame_rate=args.frame_rate,
                distributed=args.distributed,
                epoch_length=train_epoch_length,
                color_jitter=_color_jitter,
                random_resized_crop_scale=_rrc_scale,
            )
        # マルチクリップ評価: 全ランクでシャードを分割して並列評価
        data_loader_test = create_webdataset_multiclip_loader(
            shards=val_shards,
            class_to_idx=class_to_idx,
            batch_size=args.batch_size,
            num_workers=args.workers,
            crop_size=val_crop_size,
            frames_per_clip=args.clip_len,
            frame_rate=args.frame_rate,
            distributed=args.distributed,
        )

        # WebDatasetはlen()未サポートのため、概算値を使用
        dataset_classes = classes

    else:
        # mp4 モード (従来の動作)
        val_resize_size = tuple(args.val_resize_size)
        val_dir = os.path.join(args.data_path, "val")

        if not args.test_only:
            train_resize_size = tuple(args.train_resize_size)
            train_dir = os.path.join(args.data_path, "train")

            print("Loading training data")
            st = time.time()
            cache_path = _get_cache_path(train_dir, args)
            transform_train = presets.VideoClassificationPresetTrain(
                crop_size=train_crop_size, resize_size=train_resize_size
            )

            if args.cache_dataset and os.path.exists(cache_path):
                print(f"Loading dataset_train from {cache_path}")
                dataset, _ = torch.load(cache_path, weights_only=False)
                dataset.transform = transform_train
            else:
                if args.distributed:
                    print(
                        "It is recommended to pre-compute the dataset cache on a single-gpu first, as it will be faster"
                    )
                dataset = KineticsWithVideoId(
                    args.data_path,
                    frames_per_clip=args.clip_len,
                    num_classes=args.kinetics_version,
                    split="train",
                    step_between_clips=1,
                    transform=transform_train,
                    frame_rate=args.frame_rate,
                    extensions=(
                        "avi",
                        "mp4",
                    ),
                    output_format="TCHW",
                )
                if args.cache_dataset:
                    print(f"Saving dataset_train to {cache_path}")
                    utils.mkdir(os.path.dirname(cache_path))
                    utils.save_on_master((dataset, train_dir), cache_path)

            print("Took", time.time() - st)

        print("Loading validation data")
        cache_path = _get_cache_path(val_dir, args)

        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            transform_test = weights.transforms()
        else:
            transform_test = presets.VideoClassificationPresetEval(crop_size=val_crop_size, resize_size=val_resize_size)

        if args.cache_dataset and os.path.exists(cache_path):
            print(f"Loading dataset_test from {cache_path}")
            dataset_test, _ = torch.load(cache_path, weights_only=False)
            dataset_test.transform = transform_test
        else:
            if args.distributed:
                print("It is recommended to pre-compute the dataset cache on a single-gpu first, as it will be faster")
            dataset_test = KineticsWithVideoId(
                args.data_path,
                frames_per_clip=args.clip_len,
                num_classes=args.kinetics_version,
                split="val",  # "test",#
                step_between_clips=1,
                transform=transform_test,
                frame_rate=args.frame_rate,
                extensions=(
                    "avi",
                    "mp4",
                ),
                output_format="TCHW",
            )
            if args.cache_dataset:
                print(f"Saving dataset_test to {cache_path}")
                utils.mkdir(os.path.dirname(cache_path))
                utils.save_on_master((dataset_test, val_dir), cache_path)

        print("Creating data loaders")
        if not args.test_only:
            print("Found", len(dataset), "videos in training dataset")
        print("Val samples:", len(dataset_test))
        if not args.test_only:
            train_sampler = RandomClipSampler(dataset.video_clips, args.clips_per_video)
        test_sampler = UniformClipSampler(dataset_test.video_clips, args.clips_per_video)
        if args.distributed:
            if not args.test_only:
                train_sampler = DistributedSampler(train_sampler)
            test_sampler = DistributedSampler(test_sampler, shuffle=False)

        if not args.test_only:
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        num_classes = len(dataset_test.classes)
        dataset_classes = dataset_test.classes

    print("Creating model")
    # num_classes は上で設定済み
    from models import build_model, freeze_backbone, is_pytorchvideo_model

    pretrained = is_pytorchvideo_model(args.model) and args.weights == "pretrained"
    weights = None if pretrained else args.weights
    model = build_model(args.model, num_classes, pretrained=pretrained, weights=weights)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # layer freezing (config で無効化可能、v4 での全層 fine-tune 用)
    if getattr(args, "freeze_backbone", True):
        freeze_backbone(model, args.model)
    else:
        logger.info("Backbone NOT frozen: all parameters are trainable")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=getattr(args, "label_smoothing", 0.0))

    # MixUp / CutMix
    # alpha=0.0 で各々無効。両方 0.0 なら mixup_cutmix=None (適用なし)。
    # CutMix は空間領域を矩形で置換するため、動画の時間的一貫性を破壊しうる。
    # 有効にする場合は小さい alpha (例: 0.2) から試すこと。
    mixup_cutmix = None
    mixup_alpha = getattr(args, "mixup_alpha", 0.0)
    cutmix_alpha = getattr(args, "cutmix_alpha", 0.0)
    if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
        from torchvision.transforms import v2

        transforms_list = []
        if mixup_alpha > 0.0:
            transforms_list.append(v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
        if cutmix_alpha > 0.0:
            transforms_list.append(v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
        mixup_cutmix = v2.RandomChoice(transforms_list)
        logger.info(f"MixUp/CutMix enabled: mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    if args.data_source == "webdataset":
        # WebDatasetはlen()をサポートしないため、サンプル数から推定
        iters_per_epoch = args.webdataset_samples // (args.batch_size * max(1, args.world_size))
        # val 全体を評価 (切り上げで全サンプルをカバー)
        val_iters = math.ceil(val_epoch_length / args.batch_size)
        logger.info(f"Train iters/epoch: {iters_per_epoch}, Val iters: {val_iters} (samples: {val_samples})")
    else:
        iters_per_epoch = 1 if args.test_only else len(data_loader)
        val_iters = None  # mp4モードはlen()サポート
    lr_scheduler_type = getattr(args, "lr_scheduler", "multistep").lower()
    if lr_scheduler_type == "cosine":
        # CosineAnnealingLR: warmup 後の残りイテレーションで cosine decay
        cosine_iters = iters_per_epoch * (args.epochs - args.lr_warmup_epochs)
        eta_min = getattr(args, "lr_eta_min", 1e-6)
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_iters, eta_min=eta_min)
        logger.info(f"LR scheduler: CosineAnnealing (T_max={cosine_iters}, eta_min={eta_min})")
    else:
        lr_milestones = [iters_per_epoch * (m - args.lr_warmup_epochs) for m in args.lr_milestones]
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=args.lr_gamma
        )
        logger.info(f"LR scheduler: MultiStepLR (milestones={args.lr_milestones})")

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[warmup_iters],
        )
    else:
        lr_scheduler = main_lr_scheduler

    # EMA (Exponential Moving Average)
    ema_model = None
    if getattr(args, "ema_enabled", False):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        ema_decay = getattr(args, "ema_decay", 0.999)
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        logger.info(f"EMA enabled: decay={ema_decay}")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # We set weights_only to False because True gave error on cached dataset
    best_acc1_from_ckpt = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if getattr(args, "resume_weights_only", False):
            logger.info(
                f"resume_weights_only: loaded model weights from epoch {checkpoint.get('epoch', '?')}"
                f" (val_acc1={checkpoint.get('val_acc1', 0.0):.3f}). Optimizer/scheduler/epoch reset."
            )
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            best_acc1_from_ckpt = checkpoint.get("val_acc1", 0.0)
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])
            logger.info(f"Resumed from epoch {args.start_epoch - 1}, best_acc1 = {best_acc1_from_ckpt:.3f}")

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if args.data_source == "webdataset":
            _eval_model = model_without_ddp if args.distributed else model
            evaluate_webdataset_multiclip(
                _eval_model, data_loader_test, device=device, distributed=args.distributed,
            )
        else:
            evaluate(model, criterion, data_loader_test, num_classes, device=device)
        return

    logger.info("Start training")
    logger.info(f"Num of Classes: {num_classes}")
    start_time = time.time()
    best_acc1 = best_acc1_from_ckpt if args.resume else 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and args.data_source == "mp4":
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            data_loader,
            device,
            epoch,
            args.print_freq,
            scaler,
            total_iters=iters_per_epoch if args.data_source == "webdataset" else None,
            mixup_cutmix=mixup_cutmix,
            ema_model=ema_model,
        )

        if args.data_source == "webdataset":
            _eval_model = model_without_ddp if args.distributed else model
            val_acc1 = evaluate_webdataset_multiclip(
                _eval_model, data_loader_test, device=device, distributed=args.distributed,
            )
        else:
            val_acc1 = evaluate(model, criterion, data_loader_test, num_classes, device=device)

        # EMA モデルでも評価
        ema_val_acc1 = None
        if ema_model is not None:
            if args.data_source == "webdataset":
                ema_val_acc1 = evaluate_webdataset_multiclip(
                    ema_model, data_loader_test, device=device, distributed=args.distributed,
                )
            else:
                ema_val_acc1 = evaluate(ema_model, criterion, data_loader_test, num_classes, device=device)
            logger.info(f"Epoch {epoch}: EMA val_acc1 = {ema_val_acc1:.3f} (base = {val_acc1:.3f})")

        # メトリクスログ
        metrics = {"val_acc1": val_acc1}
        if ema_val_acc1 is not None:
            metrics["ema_val_acc1"] = ema_val_acc1
        log_metrics(logger, epoch, metrics)

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "val_acc1": val_acc1,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            if ema_model is not None:
                checkpoint["ema_model"] = ema_model.state_dict()

            # latest.pth を常に保存
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "latest.pth"))

            # best.pth は最良時のみ保存 (旧ファイルは削除してリネーム)
            # EMA と通常モデルの良い方で判定
            effective_acc1 = max(val_acc1, ema_val_acc1) if ema_val_acc1 is not None else val_acc1
            if effective_acc1 > best_acc1:
                best_acc1 = effective_acc1
                best_name = f"best_ep{epoch:02d}_val1_{effective_acc1:.3f}.pth".replace(".", "_", 1)
                # best_ep52_val1_84_046.pth のような形式 (小数点はアンダースコアに)
                # DDP時は master のみファイル操作 (他ランクが先に削除するとFileNotFoundError)
                if not args.distributed or args.rank == 0:
                    import glob as _glob

                    for old in _glob.glob(os.path.join(args.output_dir, "best_ep*_val1_*.pth")):
                        os.remove(old)
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, best_name))
                logger.info(f"Epoch {epoch}: New best val_acc1 = {effective_acc1:.3f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Best val_acc1: {best_acc1:.3f}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Video Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="./full_dataset/", type=str, help="dataset path (mp4モード用)")
    parser.add_argument(
        "--data-source",
        default="mp4",
        type=str,
        choices=["mp4", "webdataset"],
        help="データソース: mp4 (動画直接読込) or webdataset (事前抽出フレーム)",
    )
    parser.add_argument(
        "--webdataset-path",
        default="",
        type=str,
        help="WebDatasetディレクトリ (webdatasetモード用, 例: data/qevd/processed/webdataset)",
    )
    parser.add_argument(
        "--webdataset-samples",
        default=190000,
        type=int,
        help="WebDatasetの学習サンプル数 (LRスケジューラ用、概算でOK)",
    )
    parser.add_argument(
        "--webdataset-val-samples",
        default=0,
        type=int,
        help="WebDatasetの検証サンプル数 (0の場合はmeta.txtから自動取得)",
    )
    parser.add_argument(
        "--kinetics-version",
        default="400",
        type=str,
        help="Select kinetics version (e.g. 400, 600 for Kinetics; ignored for local datasets)",
    )
    parser.add_argument("--model", default="r2plus1d_18", type=str, help="model name")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument("--clip-len", default=8, type=int, metavar="N", help="number of frames per clip")
    parser.add_argument("--frame-rate", default=4, type=int, metavar="N", help="the frame rate")
    parser.add_argument(
        "--clips-per-video",
        default=1,
        type=int,
        metavar="N",
        help="maximum number of clips per video to consider",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=24,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=15,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=10,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 10)",
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-milestones",
        nargs="+",
        default=[20, 30, 40],
        type=int,
        help="decrease lr on milestones",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=10,
        type=int,
        help="the number of epochs to warmup (default: 10)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="linear",
        type=str,
        help="the warmup method (default: linear)",
    )
    parser.add_argument("--lr-warmup-decay", default=0.001, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )

    parser.add_argument(
        "--val-resize-size",
        default=(128, 171),
        nargs="+",
        type=int,
        help="the resize size used for validation (default: (128, 171))",
    )
    parser.add_argument(
        "--val-crop-size",
        default=(112, 112),
        nargs="+",
        type=int,
        help="the central crop size used for validation (default: (112, 112))",
    )
    parser.add_argument(
        "--train-resize-size",
        default=(128, 171),
        nargs="+",
        type=int,
        help="the resize size used for training (default: (128, 171))",
    )
    parser.add_argument(
        "--train-crop-size",
        default=(112, 112),
        nargs="+",
        type=int,
        help="the random crop size used for training (default: (112, 112))",
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )

    # Label smoothing / MixUp / CutMix
    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        help="label smoothing factor (default: 0.0)",
    )
    parser.add_argument(
        "--mixup-alpha",
        default=0.0,
        type=float,
        help="MixUp alpha (0.0 = disabled, default: 0.0)",
    )
    parser.add_argument(
        "--cutmix-alpha",
        default=0.0,
        type=float,
        help="CutMix alpha (0.0 = disabled, default: 0.0)",
    )

    # LR scheduler
    parser.add_argument(
        "--lr-scheduler",
        default="multistep",
        type=str,
        choices=["multistep", "cosine"],
        help="LR scheduler type (default: multistep)",
    )
    parser.add_argument(
        "--lr-eta-min",
        default=1e-6,
        type=float,
        help="minimum LR for CosineAnnealingLR (default: 1e-6)",
    )

    # EMA
    parser.add_argument(
        "--ema-enabled",
        action="store_true",
        help="Enable Exponential Moving Average of model weights",
    )
    parser.add_argument(
        "--ema-decay",
        default=0.999,
        type=float,
        help="EMA decay factor (default: 0.999)",
    )

    # Augmentation
    parser.add_argument(
        "--color-jitter",
        nargs="+",
        default=None,
        type=float,
        help="ColorJitter params: brightness contrast saturation [hue] (default: disabled)",
    )
    parser.add_argument(
        "--random-resized-crop-scale",
        nargs=2,
        default=None,
        type=float,
        metavar=("MIN", "MAX"),
        help="RandomResizedCrop scale range (default: disabled, use RandomCrop)",
    )

    # Backbone freezing
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze backbone layers (default: True). Use --no-freeze-backbone to unfreeze.",
    )

    # Resume options
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="When resuming, load model weights only (reset optimizer/scheduler/epoch).",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
