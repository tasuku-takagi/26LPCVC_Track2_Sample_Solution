# LPCVC-Track-2 :movie_camera: :zap:

This repository contains the solution for LPCVC 2025 Track 2: Video Classification with Dynamic Frame Selection. Our approach modifies PyTorch Vision's video classification pipeline to handle the QEVD dataset with optimized frame sampling.

## :fire: Overview

- Modified `pytorch/vision` for dynamic frame selection
- Implemented dataset preprocessing utilities for QEVD
- Added video validation and corruption checking tools
- Trained sample solution

---

## :rocket: Sample Solution

Try out the sample solution consisting of our most recent model checkpoint [here](https://drive.google.com/file/d/1vAJdpMRdJZPOPSkcVDyKu2MXOXb8qyiS/view?usp=drive_link). Read about training and evaluating the solution in the steps below.

---

## 0. Environment Setup :wrench:

### :point_right: Prerequisites

Make sure you have Python 3.10 or higher installed on your system. This repository was tested with 3.10.

### Optional (recommended): Create a conda Environment

You can use a conda environment to avoid dependency conflicts for the Track 2 repository. Create and activate one with the following:

Please ensure that you have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) downloaded.

To verify installation, open Anaconda Prompt or Terminal through Start Menu and type this:

```bash
conda --version
```

Create a new conda environment by running the following command, naming the environment whatever you'd like:

```bash
conda create --name lpcvc python=3.10 -y
```

Activate the conda environment that you crated by running the following command:

```bash
conda activate lpcvc
```

If you ever need to deactivate from the conda environment, you can run the following command:

```bash
conda deactivate
```

### Install Dependencies

Run this command to install the Python dependencies regardless if you are in an enviroment

```bash
pip install -r requirements.txt
```

This will install all the packages used for data preprocessing, model training, and video processing.

---

## 1. Modified Files :memo:

We modified the following files from the `pytorch/vision` repository:


| File                                       | Description                                |
| -------------------------------------------- | -------------------------------------------- |
| `references/video_classification/train.py` | Training script with custom configurations |
| `torchvision/datasets/video_utils.py`      | Dynamic frame selection implementation     |

---

## 2. Dataset Preprocessing :open_file_folder:

### :file_folder: Downloading the QEVD Dataset

To download the QEVD dataset, please refer to the instructions for Qualcomm's [QEVD dataset](https://www.qualcomm.com/developer/software/qevd-dataset) link.

The class labels can be found [here](class_labels.json)

### :point_right: Refactoring the QEVD Dataset

We use `refactor_dataset.py` to organize the QEVD dataset into the required directory structure.

```python
# Example usage in refactor_dataset.py
srcs = []
srcs.append(Path('./dataset/QEVD-FIT-300k-Part-1'))
srcs.append(Path('./dataset/QEVD-FIT-300k-Part-2'))
srcs.append(Path('./dataset/QEVD-FIT-300k-Part-3'))
srcs.append(Path('./dataset/QEVD-FIT-300k-Part-4'))
dest = Path('./QEVD_organized')

refactor = DatasetRefactorer(srcs, dest, Path('./dataset/fine_grained_labels_release.json'))
refactor.refactor_dataset()
```

The script organizes videos into the following structure:

```
root/
├── train/
│   ├── action_category_1/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   └── action_category_2/
└── val/
    ├── action_category_1/
    └── action_category_2/
```

### :broom: Cleaning Corrupted Videos

We use `check_videos.py` to validate and clean corrupted videos from the dataset.

```bash
# Check and quarantine corrupted videos
python check_videos.py --root ./full_dataset \
    --quarantine ./bad_videos \
    --remux \
    --replace \
    --ext mp4 \
    --jobs 8 \
    --report bad_videos.csv
```

---

## 3. Dynamic Frame Selection :film_strip:

### :bulb: Key Modification

We modified `compute_clips_for_video()` in `torchvision/datasets/video_utils.py` to implement dynamic frame selection for videos with fewer frames than required.

```python
# Dynamic frame rate adjustment in video_utils.py
if total_frames < num_frames:
    # Calculate video duration
    video_duration = len(video_pts) / fps

    total_frames = num_frames
    frame_rate = math.ceil(num_frames / video_duration)
```

This ensures that short videos are properly sampled by dynamically adjusting the frame rate.

---

## 4. Training the Model :weight_lifting:

### :point_right: Training Command

Run the training script with the following arguments:

```bash
python references/video_classification/train.py \
    --data-path /path/to/dataset/root \
    --weights KINETICS400_V1 \
    --cache-dataset \
    --epochs 15 \
    --batch-size 24 \
    --lr 0.01
```

### :gear: Key Parameters


| Parameter         | Description                                                    | Example            |
| ------------------- | ---------------------------------------------------------------- | -------------------- |
| `--data-path`     | Path to dataset root (`root/train` or `val/action_categories`) | `./full_dataset/`  |
| `--resume`        | Path to checkpoint for resuming training                       | `./checkpoint.pth` |
| `--start-epoch`   | Starting epoch when resuming from checkpoint                   | `10`               |
| `--weights`       | Pre-trained weights to use                                     | `KINETICS400_V1`   |
| `--cache-dataset` | Cache processed dataset for faster loading                     | (flag)             |
| `--batch-size`    | Batch size per GPU                                             | `24`               |
| `--epochs`        | Total number of epochs                                         | `15`               |
| `--lr`            | Initial learning rate                                          | `0.01`             |

### :warning: Important Notes

- For distributed training, it's recommended to pre-compute the dataset cache on a single GPU first
- The model uses `r2plus1d_18` architecture by default
- Layer freezing is enabled for faster training (only `layer4` and `fc` are trainable)

**More details on parameters can be found in the `get_args_parser()` function in `train.py`**

---

## 5. Model Validation :white_check_mark:

### :point_right: Validation Command

To validate a trained model, run:

```bash
python references/video_classification/train.py \
    --data-path /path/to/dataset/root \
    --resume /path/to/checkpoint.pth \
    --test-only
```

### :bar_chart: Evaluation Metrics

The validation process reports:

- **Clip Accuracy (Acc@1, Acc@5)**: Accuracy per video clip
- **Video Accuracy (Acc@1, Acc@5)**: Aggregated accuracy across all clips of a video

```python
# Aggregation of clip predictions for video-level accuracy
preds = torch.softmax(output, dim=1)
for b in range(video.size(0)):
    idx = video_idx[b].item()
    agg_preds[idx] += preds[b].detach()
    agg_targets[idx] = target[b].detach().item()
```

---

## :link: References

- Built on [PyTorch Vision](https://github.com/pytorch/vision)
- Dataset: QEVD (Query-based Event Video Dataset)
