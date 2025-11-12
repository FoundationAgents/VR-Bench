#!/usr/bin/env python3
"""
Resize every dataset video to a uniform 192 frames (8 seconds at 24 fps) without
changing the frame rate. Uses the same traversal and retiming helpers as
normalize_videos.py but applies a fixed target instead of per-folder averages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from normalize_videos import (
    DATASET_DIFFICULTIES,
    DATASET_ROOT,
    DATASET_VARIANTS,
    DEFAULT_TARGET_FPS,
    discover_dataset_folders,
    ensure_ffmpeg_available,
    gather_metadata,
    retime_video,
)
from video_stats import VideoMetadataError, get_video_metadata, iter_videos

DEFAULT_TARGET_FRAMES = 192
REPORT_PATH = Path(__file__).with_name("scaled_videos_stats.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch rescale videos so they all contain a fixed number of frames at 24 fps "
            "(default: 192 frames)."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Single folder to process. When omitted, all matching dataset folders are processed.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Destination directory for processed videos. If omitted, videos are updated in place.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Root dataset directory for batch processing (default: preconfigured DATASET_ROOT).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process videos located in subdirectories as well.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist. In-place mode always overwrites.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended ffmpeg actions without executing them.",
    )
    parser.add_argument(
        "--retain-audio",
        action="store_true",
        help="Keep and retime audio tracks (requires ffmpeg). Without this flag audio is removed.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=DEFAULT_TARGET_FRAMES,
        help=f"Desired frame count for every video (default: {DEFAULT_TARGET_FRAMES}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        ensure_ffmpeg_available()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.target_frames <= 0:
        print("Target frame count must be a positive integer.", file=sys.stderr)
        return 1

    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else DATASET_ROOT
    target_frames = args.target_frames
    target_fps = DEFAULT_TARGET_FPS
    target_duration = target_frames / target_fps

    if args.folder:
        target_folders = [Path(args.folder).expanduser().resolve()]
    else:
        target_folders = discover_dataset_folders(
            dataset_root,
            DATASET_VARIANTS,
            DATASET_DIFFICULTIES,
        )
        if not target_folders:
            print(f"No matching video folders found under {dataset_root}")
            return 1

    output_dir_base = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    all_failures: List[Tuple[Path, str]] = []
    total_processed = 0
    total_candidates = 0
    report_data: List[Dict[str, Any]] = []

    for folder in target_folders:
        if not folder.exists() or not folder.is_dir():
            message = f"Folder not found: {folder}"
            print(message, file=sys.stderr)
            all_failures.append((folder, message))
            continue

        print(f"\n=== Processing folder: {folder} ===")
        videos = list(iter_videos(folder, args.recursive))
        if not videos:
            print("No videos found in this folder.")
            continue

        metadata_map, metadata_errors = gather_metadata(videos)
        if not metadata_map:
            print("No analyzable videos in this folder; skipping.")
            for video, msg in metadata_errors:
                all_failures.append((video, msg))
            continue

        if metadata_errors:
            print("Metadata issues encountered:")
            for video, msg in metadata_errors:
                print(f"- {video}: {msg}")
            all_failures.extend(metadata_errors)

        print(
            f"Targeting {target_frames} frames @ {target_fps} fps (~{target_duration:.3f}s) "
            f"for {len(metadata_map)} video(s)."
        )

        if output_dir_base is not None:
            try:
                relative = folder.relative_to(dataset_root)
            except ValueError:
                relative = Path(folder.name)
            folder_output_dir = (output_dir_base / relative).resolve()
        else:
            folder_output_dir = None

        analyzable_videos = list(metadata_map.keys())
        total_candidates += len(analyzable_videos)
        folder_successes: List[Tuple[Path, Path, float]] = []
        folder_failures: List[Tuple[Path, str]] = []
        folder_report: Dict[str, Any] = {
            "folder": str(folder),
            "target_duration": target_duration,
            "target_fps": target_fps,
            "target_frames": target_frames,
            "videos": [],
        }

        for index, video in enumerate(analyzable_videos, start=1):
            print(f"[{index}/{len(analyzable_videos)}] Processing {video} ...")
            try:
                source, dest, factor = retime_video(
                    video,
                    folder_output_dir,
                    args.overwrite,
                    args.dry_run,
                    retain_audio=args.retain_audio,
                    target_duration=target_duration,
                    target_fps=target_fps,
                    precomputed_metadata=metadata_map.get(video),
                )
            except RuntimeError as exc:
                message = str(exc)
                folder_failures.append((video, message))
                print(f"- Failed: {video}: {message}")
                continue

            folder_successes.append((source, dest, factor))

            if args.dry_run:
                print(f"Dry run: would write {dest} (speed factor {factor:.4f})")
                duration_before, frames_before, _ = metadata_map[video]
                folder_report["videos"].append(
                    {
                        "video": str(dest),
                        "speed_factor": factor,
                        "duration": duration_before,
                        "frames": frames_before,
                        "note": "dry-run; duration/frames reflect original values",
                    }
                )
                continue

            try:
                duration_after, frames_after, _ = get_video_metadata(dest)
            except VideoMetadataError as exc:
                message = f"Post-processing metadata read failed: {exc}"
                folder_failures.append((dest, message))
                print(f"- {dest}: {message}")
                continue

            folder_report["videos"].append(
                {
                    "video": str(dest),
                    "speed_factor": factor,
                    "duration": duration_after,
                    "frames": frames_after,
                }
            )

        total_processed += len(folder_successes)

        if folder_failures:
            print("Encountered issues:")
            for video, message in folder_failures:
                print(f"- {video}: {message}")
            all_failures.extend(folder_failures)

        print(f"Folder summary: processed {len(folder_successes)} of {len(analyzable_videos)} analyzed video(s).")

        if folder_report["videos"]:
            report_data.append(folder_report)

    if report_data:
        try:
            with REPORT_PATH.open("w", encoding="utf-8") as handle:
                json.dump(report_data, handle, indent=2)
            print(f"\nSaved report to {REPORT_PATH}")
        except OSError as exc:
            message = f"Failed to write report: {exc}"
            print(message, file=sys.stderr)
            all_failures.append((REPORT_PATH, message))
    else:
        print("\nNo report generated (no processed videos).")

    print(
        f"\nOverall processed {total_processed} of {total_candidates} analyzed video(s) "
        f"across {len(report_data)} folder(s) with reports."
    )

    if all_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
