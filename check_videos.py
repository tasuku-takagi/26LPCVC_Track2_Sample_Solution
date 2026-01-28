#!/usr/bin/env python3
"""
Check, optionally remux, and quarantine corrupted videos in a dataset organized as:

full_dataset/
  train/
    action_label_1/
      vid1.mp4
      vid2.mp4
    action_label_2/
  val/
    action_label_1/
    ...

Features:
- Walks only `train` and `val` subfolders under the provided root.
- Probes each video using PyAV if available, otherwise falls back to an ffmpeg probe.
- Optionally attempts a remux with ffmpeg for files that fail to probe.
- Moves still-broken files into a quarantine directory that preserves the original relative structure:
    <quarantine>/<split>/<label>/filename.mp4
- Can replace original with remuxed file (--replace) or keep remuxed alongside original.
- Produces an optional CSV report listing all checked files and their status.

Usage example:
  python scripts/check_videos.py --root /path/to/full_dataset \
      --quarantine /tmp/bad_videos --remux --replace --ext mp4 --jobs 8 --report bad_videos.csv

Notes:
- Requires ffmpeg in PATH for remux/probe fallback.
- If using virtualenv/conda, run this in the same environment you use for training so the same `av` package is used.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import av  # PyAV
except Exception:
    av = None  # we'll fall back to ffmpeg probe if necessary


def find_videos_under_splits(root: Path, exts: List[str]) -> List[Path]:
    """Find video files only under train/ and val/ splits, preserving action-label dirs."""
    videos = []
    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue
            for p in label_dir.rglob("*"):
                if p.is_file() and any(p.name.lower().endswith("." + e.lower()) for e in exts):
                    videos.append(p)
    return videos


def probe_with_av(path: Path) -> bool:
    """Try to open video with PyAV to confirm it's readable."""
    try:
        container = av.open(str(path), metadata_errors="ignore")
        # Accessing streams forces parsing of container metadata
        _ = container.streams
        # optionally try reading a packet/frame quickly? just probing is enough
        container.close()
        return True
    except Exception:
        return False


def probe_with_ffmpeg(path: Path) -> bool:
    """Use ffmpeg as a fallback to probe file readability."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return p.returncode == 0
    except FileNotFoundError:
        # ffmpeg not available
        return False


def is_video_ok(path: Path) -> bool:
    if av is not None:
        ok = probe_with_av(path)
        if ok:
            return True
        # If av exists but failed, still allow ffmpeg fallback to catch different errors
        return probe_with_ffmpeg(path)
    else:
        return probe_with_ffmpeg(path)


def remux_with_ffmpeg(path: Path, out_path: Path) -> bool:
    """Attempt to remux the file with ffmpeg by copying streams."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-c", "copy", str(out_path)]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return p.returncode == 0
    except FileNotFoundError:
        return False


def ensure_parent_exists(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def quarantine_path_for(root: Path, root_base: Path, quarantine_root: Path, file_path: Path) -> Path:
    """
    Preserve dataset structure under quarantine_root.
    Example:
      root_base = /path/to/full_dataset
      file_path = /path/to/full_dataset/train/toe_touch/0001.mp4
      returns quarantine_root/train/toe_touch/0001.mp4
    """
    rel = file_path.relative_to(root_base)
    dest = quarantine_root / rel
    return dest


def process_video(
    path: Path,
    root_base: Path,
    quarantine_root: Path,
    remux: bool,
    replace: bool,
    dry_run: bool,
) -> Tuple[Path, str, Optional[str]]:
    """
    Returns (path, status, notes)
    status: "ok", "remuxed_replaced", "remuxed_kept", "quarantined", "remux_failed_quarantined", "probe_failed_quarantined"
    """
    try:
        if is_video_ok(path):
            return path, "ok", None

        # Not ok -> attempt remux if requested
        if remux:
            tmp_out = path.with_suffix(path.suffix + ".remux.mp4")
            if dry_run:
                return path, "would_remux_and_replace" if replace else "would_remux_keep", None

            remux_ok = remux_with_ffmpeg(path, tmp_out)
            if remux_ok and is_video_ok(tmp_out):
                if replace:
                    try:
                        # atomic replace
                        tmp_replacement = path.with_suffix(path.suffix + ".bak")
                        os.replace(path, tmp_replacement)  # move original to .bak (platform dependent atomicity)
                        os.replace(tmp_out, path)  # move remux into original path
                        try:
                            os.remove(tmp_replacement)
                        except Exception:
                            # leave the .bak if removal fails
                            pass
                        return path, "remuxed_replaced", None
                    except Exception as e:
                        # fallback: keep remuxed file next to original
                        new_dest = path.with_name("remuxed_" + path.name)
                        shutil.move(str(tmp_out), str(new_dest))
                        return path, "remuxed_kept", f"replaced_failed:{e}"
                else:
                    # keep remuxed file next to original
                    new_dest = path.with_name("remuxed_" + path.name)
                    shutil.move(str(tmp_out), str(new_dest))
                    return path, "remuxed_kept", str(new_dest)
            else:
                # remux failed or produced broken file -> quarantine original
                dest = quarantine_path_for(root_base, root_base, quarantine_root, path)
                if dry_run:
                    return path, "would_quarantine_after_remux_failed", None
                ensure_parent_exists(dest)
                shutil.move(str(path), str(dest))
                return path, "remux_failed_quarantined", str(dest)
        else:
            # No remux requested: quarantine original
            dest = quarantine_path_for(root_base, root_base, quarantine_root, path)
            if dry_run:
                return path, "would_quarantine", None
            ensure_parent_exists(dest)
            shutil.move(str(path), str(dest))
            return path, "quarantined", str(dest)
    except Exception as e:
        # Catch-all: attempt to quarantine to avoid leaving broken files in place
        try:
            dest = quarantine_path_for(root_base, root_base, quarantine_root, path)
            if not dry_run:
                ensure_parent_exists(dest)
                shutil.move(str(path), str(dest))
            return path, "error_quarantined", f"{e}"
        except Exception as e2:
            return path, "fatal_error", f"{e} | quarantine_failed:{e2}"


def write_report(rows: List[Tuple[str, str, str]], report_path: Path):
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "status", "notes"])
        for r in rows:
            writer.writerow(r)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Validate/remux/quarantine videos under full_dataset/train and full_dataset/val")
    parser.add_argument("--root", type=Path, default=Path("full_dataset"), help="Root dataset directory containing train/ and val/")
    parser.add_argument("--ext", nargs="+", default=["mp4"], help="Video extensions to check (default: mp4)")
    parser.add_argument("--quarantine", type=Path, required=True, help="Directory to move bad videos into")
    parser.add_argument("--remux", action="store_true", help="Try to remux broken files with ffmpeg before quarantining")
    parser.add_argument("--replace", action="store_true", help="When remux succeeds, replace original files with remuxed ones (default: keep remuxed as remuxed_<name>)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)), help="Number of worker threads for probing (default: CPU count)")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually move or remux anything; just report what would be done")
    parser.add_argument("--report", type=Path, default=None, help="CSV path to save report (optional)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    quarantine_root = args.quarantine.resolve()

    if not root.exists():
        print(f"Root path {root} does not exist.", file=sys.stderr)
        sys.exit(2)

    quarantine_root.mkdir(parents=True, exist_ok=True)

    exts = args.ext
    print(f"Scanning under {root} for extensions: {exts}")
    videos = find_videos_under_splits(root, exts)
    print(f"Found {len(videos)} videos to check (under train/ and val/).")

    if len(videos) == 0:
        print("No videos found. Exiting.")
        return

    rows = []
    process = partial(
        process_video,
        root_base=root,
        quarantine_root=quarantine_root,
        remux=args.remux,
        replace=args.replace,
        dry_run=args.dry_run,
    )

    print(f"Probing videos with {'PyAV' if av is not None else 'ffmpeg (fallback)'}; jobs={args.jobs}; remux={args.remux}; replace={args.replace}; dry_run={args.dry_run}")

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(process, p): p for p in videos}
        done = 0
        bad_count = 0
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                path, status, notes = fut.result()
            except Exception as e:
                path, status, notes = p, "worker_exception", str(e)
            done += 1
            if status != "ok":
                bad_count += 1
            rows.append((str(path), status, "" if notes is None else str(notes)))
            if done % 50 == 0 or done == len(videos):
                print(f"Progress: {done}/{len(videos)} checked; bad so far: {bad_count}")

    print(f"Finished. Total videos: {len(videos)}; problematic: {bad_count}")
    if args.report:
        write_report(rows, args.report)
        print(f"Wrote report to {args.report}")

    # Print some examples
    examples = [r for r in rows if r[1] != "ok"]
    if examples:
        print("Examples of problematic files (up to 20):")
        for ex_row in examples[:20]:
            print(" ", ex_row)

    print("Done.")


if __name__ == "__main__":
    main()
