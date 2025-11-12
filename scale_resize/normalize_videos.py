#!/usr/bin/env python3
"""
Retiming helper to normalize multiple video folders based on their average clip
length. For each folder, the script computes the mean duration, rounds it to the
nearest 0.5 s boundary, and re-encodes every clip so it matches that duration at
24 fps. Audio tracks are tempo-adjusted when present so that the clip stays in
sync with the retimed video.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from video_stats import VideoMetadataError, get_video_metadata, iter_videos

DEFAULT_TARGET_FPS = 24
DATASET_ROOT = Path(r"C:\Users\Administrator\Desktop\dataset_VR")
DATASET_VARIANTS = ["irregular_maze", "maze", "sokoban", "trapfield", "maze3d"]
DATASET_DIFFICULTIES = ["easy", "medium", "hard"]
REPORT_PATH = Path(__file__).with_name("normalized_videos_stats.json")


def ensure_ffmpeg_available() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required but not found in PATH.")
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe (ships with ffmpeg) is required but not found in PATH.")


def has_audio_stream(path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())


def atempo_filters(speed: float) -> str | None:
    if math.isclose(speed, 1.0, rel_tol=0.0, abs_tol=1e-3):
        return None

    filters: List[str] = []
    remaining = speed

    while remaining > 2.0 + 1e-6:
        filters.append("atempo=2.0")
        remaining /= 2.0

    while remaining < 0.5 - 1e-6:
        filters.append("atempo=0.5")
        remaining /= 0.5

    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def round_to_half(value: float) -> float:
    if math.isinf(value) or math.isnan(value):
        raise ValueError("Cannot round non-finite value.")
    return round(value * 2.0) / 2.0


def discover_dataset_folders(
    base_root: Path,
    variants: Iterable[str],
    difficulties: Iterable[str],
) -> List[Path]:
    folders: List[Path] = []
    for variant in variants:
        variant_dir = base_root / variant
        if not variant_dir.exists() or not variant_dir.is_dir():
            continue

        numeric_dirs = sorted(
            (child for child in variant_dir.iterdir() if child.is_dir() and child.name.isdigit()),
            key=lambda child: int(child.name),
        )
        for numeric_dir in numeric_dirs:
            for difficulty in difficulties:
                videos_dir = numeric_dir / difficulty / "videos"
                if videos_dir.is_dir():
                    folders.append(videos_dir.resolve())
    return folders


def gather_metadata(videos: Iterable[Path]) -> Tuple[Dict[Path, Tuple[float, float, float]], List[Tuple[Path, str]]]:
    metadata: Dict[Path, Tuple[float, float, float]] = {}
    errors: List[Tuple[Path, str]] = []
    for video in videos:
        try:
            metadata[video] = get_video_metadata(video)
        except VideoMetadataError as exc:
            errors.append((video, str(exc)))
    return metadata, errors


def retime_video(
    path: Path,
    output_dir: Path | None,
    overwrite: bool,
    dry_run: bool,
    retain_audio: bool,
    target_duration: float,
    target_fps: float,
    precomputed_metadata: Tuple[float, float, float] | None = None,
) -> Tuple[Path, Path, float]:
    if precomputed_metadata is not None:
        duration_s = precomputed_metadata[0]
    else:
        try:
            duration_s, _, _ = get_video_metadata(path)
        except VideoMetadataError as exc:
            raise RuntimeError(f"Cannot read metadata for {path}: {exc}") from exc

    video_factor = target_duration / duration_s
    audio_speed = duration_s / target_duration

    vf = f"setpts={video_factor:.12f}*PTS,fps={target_fps}"
    af = atempo_filters(audio_speed) if retain_audio and has_audio_stream(path) else None

    overwrite_output = overwrite or output_dir is None

    if output_dir is None:
        temp_output_path = path.with_name(f".__normalized__{uuid.uuid4().hex}{path.suffix}")
        final_output_path = path
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_output_path = output_dir / path.name
        final_output_path = temp_output_path

    cmd: List[str] = ["ffmpeg", "-y" if overwrite_output else "-n", "-i", str(path), "-vf", vf]
    if retain_audio:
        if af:
            cmd.extend(["-af", af])
    else:
        cmd.append("-an")
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if retain_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])

    cmd.extend(["-r", f"{target_fps:.6g}", str(temp_output_path)])

    if dry_run:
        return path, final_output_path, video_factor

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if output_dir is None:
            with contextlib.suppress(FileNotFoundError, PermissionError):
                temp_output_path.unlink()
        raise RuntimeError(result.stderr.strip() or f"ffmpeg failed on {path}")

    if output_dir is None:
        try:
            temp_output_path.replace(path)
        finally:
            with contextlib.suppress(FileNotFoundError, PermissionError):
                temp_output_path.unlink()

    return path, final_output_path, video_factor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch retime videos so each folder matches the nearest 0.5 second average duration at 24 fps."
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        ensure_ffmpeg_available()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else DATASET_ROOT

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

        avg_duration = sum(meta[0] for meta in metadata_map.values()) / len(metadata_map)
        target_duration = round_to_half(avg_duration)
        target_fps = DEFAULT_TARGET_FPS
        target_frames = int(round(target_duration * target_fps))

        print(
            f"Average duration {avg_duration:.3f}s -> target {target_duration:.1f}s "
            f"({target_frames} frames @ {target_fps} fps)."
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
            "average_duration": avg_duration,
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

        print(
            f"Folder summary: processed {len(folder_successes)} of {len(analyzable_videos)} analyzed video(s)."
        )

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
        f"\nOverall processed {total_processed} of {total_candidates} analyzed video(s) across {len(report_data)} folder(s) with reports."
    )

    if all_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
