import datetime
import io
import os
import pickle
import random
import time
import warnings

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
    mp4直接読み込みと同等の動作を再現。
    """

    def __init__(
        self,
        frames_per_clip: int = 8,
        frame_rate: int = 4,
        is_train: bool = True,
    ):
        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate
        self.is_train = is_train

    def __call__(self, frames: list[bytes]) -> list[bytes]:
        total_frames = len(frames)
        clip_len = self.frames_per_clip * self.frame_rate

        if total_frames < clip_len:
            # フレーム不足時は繰り返しでパディング
            frames = frames * ((clip_len // total_frames) + 1)
            total_frames = len(frames)

        max_start = total_frames - clip_len

        if self.is_train:
            start = random.randint(0, max_start)
        else:
            start = max_start // 2  # 中央からサンプリング

        indices = [start + i * self.frame_rate for i in range(self.frames_per_clip)]
        return [frames[i] for i in indices]


class WebDatasetFrameTransform:
    """
    WebDataset用フレーム変換。presets.VideoClassificationPreset と同等。
    """

    def __init__(
        self,
        crop_size: tuple[int, int] = (112, 112),
        is_train: bool = True,
    ):
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]

        if is_train:
            self.transform = T.Compose([
                T.RandomCrop(crop_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = T.Compose([
                T.CenterCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


def make_webdataset_transform(
    clip_sampler: WebDatasetClipSampler,
    frame_transform: WebDatasetFrameTransform,
    class_to_idx: dict[str, int],
):
    """
    WebDatasetのサンプル変換関数を作成。
    mp4版と同じ出力形式: (video, audio, label, video_idx)
    """

    def transform(sample: dict) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        # フレームをデコード
        frames_pkl = sample["frames.pkl"]
        frames: list[bytes] = pickle.loads(frames_pkl)

        # ランダムクリップをサンプリング
        sampled_frames = clip_sampler(frames)

        # 各フレームを変換
        tensors = []
        for frame_bytes in sampled_frames:
            img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            tensors.append(frame_transform(img))

        # [T, C, H, W] -> [C, T, H, W] (TCHW -> CTHW)
        video = torch.stack(tensors, dim=0)  # [T, C, H, W]
        video = video.permute(1, 0, 2, 3)    # [C, T, H, W]

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
) -> torch.utils.data.DataLoader:
    """
    WebDataset DataLoaderを作成。
    """
    if not HAS_WEBDATASET:
        raise ImportError("webdataset is required. Install with: pip install webdataset")

    clip_sampler = WebDatasetClipSampler(
        frames_per_clip=frames_per_clip,
        frame_rate=frame_rate,
        is_train=is_train,
    )
    frame_transform = WebDatasetFrameTransform(
        crop_size=crop_size,
        is_train=is_train,
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
        # Validationも分散: split_by_nodeで各rankが別シャードを処理
        # empty_check=False + num_workers=0 (後述) でworker分割問題を回避
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
):
    """
    Args:
        total_iters: 総イテレーション数 (WebDatasetなどlen()未サポートの場合に指定)
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter(
        "clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}")
    )

    header = f"Epoch: [{epoch}]"
    iteration = 0
    for video, _, target, _ in metric_logger.log_every(data_loader, print_freq, header, total=total_iters):
        start_time = time.time()
        video, target = video.to(device), target.to(device)
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

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(batch_size / (time.time() - start_time))
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

    print(
        " * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5
        )
    )
    return metric_logger.acc1.global_avg


def evaluate(model, criterion, data_loader, num_classes, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    # Group and aggregate output of a video
    num_videos = len(data_loader.dataset.samples)
    print(f"Evaluating {num_videos} videos\n")
    # num_classes = len(data_loader.dataset.classes)
    agg_preds = torch.zeros(
        (num_videos, num_classes), dtype=torch.float32, device=device
    )
    agg_targets = torch.zeros((num_videos), dtype=torch.int32, device=device)
    with torch.inference_mode():
        for video, _, target, video_idx in metric_logger.log_every(
            data_loader, 100, header
        ):
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
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the sampler has {num_data_from_sampler} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(
        " * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5
        )
    )
    # Reduce the agg_preds and agg_targets from all gpu and show result
    agg_preds = utils.reduce_across_processes(agg_preds)
    agg_targets = utils.reduce_across_processes(
        agg_targets, op=torch.distributed.ReduceOp.MAX
    )
    agg_acc1, agg_acc5 = utils.accuracy(agg_preds, agg_targets, topk=(1, 5))
    print(
        " * Video Acc@1 {acc1:.3f} Video Acc@5 {acc5:.3f}".format(
            acc1=agg_acc1, acc5=agg_acc5
        )
    )
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath, args):
    import hashlib

    value = f"{filepath}-{args.clip_len}-{args.kinetics_version}-{args.frame_rate}"
    h = hashlib.sha1(value.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def collate_fn(batch):
    # remove audio from the batch
    return default_collate(batch)


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

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
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        num_classes = len(classes)
        print(f"Classes: {num_classes}")

        # シャードパターン (存在するシャードを自動検出)
        import glob
        train_shard_files = sorted(glob.glob(os.path.join(args.webdataset_path, "train", "shard-*.tar")))
        val_shard_files = sorted(glob.glob(os.path.join(args.webdataset_path, "val", "shard-*.tar")))
        if not train_shard_files:
            raise FileNotFoundError(f"No train shards found in {args.webdataset_path}/train/")
        train_shards = train_shard_files  # リストで渡す
        val_shards = val_shard_files if val_shard_files else train_shard_files[:1]  # valがなければtrainの一部を使用
        print(f"Found {len(train_shard_files)} train shards, {len(val_shard_files)} val shards")

        # 分散学習で各rankが同じイテレーション数になるようepoch_lengthを設定
        # (NCCLデッドロック回避のため必須)
        train_epoch_length = args.webdataset_samples // max(1, args.world_size)
        val_epoch_length = max(1000, train_epoch_length // 10)  # valはtrainの約10%

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
        )
        # 分散学習のvalidationではworker=0が最も安全 (シャード分割の問題を回避)
        val_workers = 0 if args.distributed else args.workers
        data_loader_test = create_webdataset_loader(
            shards=val_shards,
            class_to_idx=class_to_idx,
            batch_size=args.batch_size,
            num_workers=val_workers,
            is_train=False,
            crop_size=val_crop_size,
            frames_per_clip=args.clip_len,
            frame_rate=args.frame_rate,
            distributed=args.distributed,
            epoch_length=val_epoch_length,
        )

        # WebDatasetはlen()未サポートのため、概算値を使用
        dataset_classes = classes

    else:
        # mp4 モード (従来の動作)
        val_resize_size = tuple(args.val_resize_size)
        train_resize_size = tuple(args.train_resize_size)

        train_dir = os.path.join(args.data_path, "train")
        val_dir = os.path.join(args.data_path, "val")

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
            transform_test = presets.VideoClassificationPresetEval(
                crop_size=val_crop_size, resize_size=val_resize_size
            )

        if args.cache_dataset and os.path.exists(cache_path):
            print(f"Loading dataset_test from {cache_path}")
            dataset_test, _ = torch.load(cache_path, weights_only=False)
            dataset_test.transform = transform_test
        else:
            if args.distributed:
                print(
                    "It is recommended to pre-compute the dataset cache on a single-gpu first, as it will be faster"
                )
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
        print("Found", len(dataset), "videos in training dataset")
        print("Val samples:", len(dataset_test))
        train_sampler = RandomClipSampler(dataset.video_clips, args.clips_per_video)
        test_sampler = UniformClipSampler(dataset_test.video_clips, args.clips_per_video)
        if args.distributed:
            train_sampler = DistributedSampler(train_sampler)
            test_sampler = DistributedSampler(test_sampler, shuffle=False)

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

        num_classes = len(dataset.classes)
        dataset_classes = dataset.classes

    print("Creating model")
    # num_classes は上で設定済み
    model = torchvision.models.get_model(args.model, weights=args.weights)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # optional layer freezing for faster training
    for name, param in model.named_parameters():
        if not name.startswith("layer4") and not name.startswith("fc"):
            param.requires_grad = False

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

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
        # valのイテレーション数 (trainの約10%と仮定)
        val_iters = max(1, iters_per_epoch // 10)
    else:
        iters_per_epoch = len(data_loader)
        val_iters = None  # mp4モードはlen()サポート
    lr_milestones = [
        iters_per_epoch * (m - args.lr_warmup_epochs) for m in args.lr_milestones
    ]
    main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma
    )

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

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # We set weights_only to False because True gave error on cached dataset
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if args.data_source == "webdataset":
            evaluate_webdataset(model, criterion, data_loader_test, device=device, total_iters=val_iters)
        else:
            evaluate(model, criterion, data_loader_test, num_classes, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and args.data_source == "mp4":
            train_sampler.set_epoch(epoch)
        print("Num of Classes", num_classes)
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
        )
        if args.data_source == "webdataset":
            evaluate_webdataset(model, criterion, data_loader_test, device=device, total_iters=val_iters)
        else:
            evaluate(model, criterion, data_loader_test, num_classes, device=device)
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth")
            )
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Video Classification Training", add_help=add_help
    )

    parser.add_argument(
        "--data-path", default="./full_dataset/", type=str, help="dataset path (mp4モード用)"
    )
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
    parser.add_argument(
        "--clip-len", default=8, type=int, metavar="N", help="number of frames per clip"
    )
    parser.add_argument(
        "--frame-rate", default=4, type=int, metavar="N", help="the frame rate"
    )
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
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
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
    parser.add_argument(
        "--lr-warmup-decay", default=0.001, type=float, help="the decay for lr"
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "--output-dir", default=".", type=str, help="path to save outputs"
    )
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
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
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
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
    parser.add_argument(
        "--weights", default=None, type=str, help="the weights enum name to load"
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
