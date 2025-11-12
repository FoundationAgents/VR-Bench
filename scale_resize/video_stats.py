#!/usr/bin/env python3
"""
Compute the average duration (seconds), frame count, and check FPS consistency
for videos in a folder.

The script attempts to read metadata via ffprobe when available, falling back to
OpenCV if the ffmpeg tools are missing. Install ffmpeg or opencv-python before
running if neither is already present in your environment.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Iterable, List, Tuple


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".mkv",
    ".avi",
    ".wmv",
    ".flv",
    ".webm",
}

DEFAULT_VIDEO_FOLDER = Path(r"C:\Users\Administrator\Desktop\dataset_VR\maze3d\1\medium\videos")


class VideoMetadataError(RuntimeError):
    """Raised when video metadata cannot be extracted."""


def iter_videos(folder: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for candidate in folder.glob(pattern):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            yield candidate


def get_video_metadata(path: Path) -> tuple[float, float, float]:
    try:
        return _metadata_with_ffprobe(path)
    except FileNotFoundError:
        pass
    except VideoMetadataError:
        raise
    except Exception as exc:
        raise VideoMetadataError(f"ffprobe error for {path}: {exc}") from exc

    try:
        return _metadata_with_cv2(path)
    except ImportError:
        raise VideoMetadataError(
            "Neither ffprobe (from ffmpeg) nor OpenCV (cv2) is available"
        )
    except Exception as exc:
        raise VideoMetadataError(f"OpenCV error for {path}: {exc}") from exc


def _metadata_with_ffprobe(path: Path) -> tuple[float, float, float]:
    """Extract duration, frame count, and FPS using ffprobe."""
    if not shutil.which("ffprobe"):
        raise FileNotFoundError("ffprobe not found")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise VideoMetadataError(result.stderr.strip() or "ffprobe failed")

    data = json.loads(result.stdout)
    try:
        duration_s = float(data["format"]["duration"])
    except (KeyError, ValueError) as exc:
        raise VideoMetadataError("Duration unavailable") from exc

    stream = data.get("streams", [{}])[0]
    raw_frames = stream.get("nb_frames")
    avg_frame_rate = stream.get("avg_frame_rate")
    fps = _fps_from_rate(avg_frame_rate)

    if raw_frames in (None, "N/A"):
        frames = _frames_from_rate(duration_s, avg_frame_rate)
    else:
        try:
            frames = float(raw_frames)
        except ValueError as exc:
            raise VideoMetadataError("Invalid frame count") from exc

    return duration_s, frames, fps


def _metadata_with_cv2(path: Path) -> tuple[float, float, float]:
    """Extract duration, frame count, and FPS using OpenCV."""
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoMetadataError("Unable to open video with OpenCV")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise VideoMetadataError("Frame rate unavailable via OpenCV")

    if not math.isfinite(frame_count) or frame_count <= 0:
        frame_count = float(_count_frames_with_cv2(capture))

    frames = frame_count
    capture.release()

    duration_s = frames / fps
    return duration_s, frames, fps


def _count_frames_with_cv2(capture) -> int:
    frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames += 1
    return frames


def _group_fps(fps_values: Iterable[float], tolerance: float = 1e-3) -> List[Tuple[float, int]]:
    groups: List[Tuple[float, int]] = []
    for fps in fps_values:
        for index, (group_fps, count) in enumerate(groups):
            if math.isclose(group_fps, fps, rel_tol=0.0, abs_tol=tolerance):
                groups[index] = (group_fps, count + 1)
                break
        else:
            groups.append((fps, 1))
    return groups


def _frames_from_rate(duration_s: float, avg_frame_rate: str | None) -> float:
    fps = _fps_from_rate(avg_frame_rate)
    return float(duration_s * fps)


def _fps_from_rate(avg_frame_rate: str | None) -> float:
    if not avg_frame_rate or avg_frame_rate in ("0/0", "0"):
        raise VideoMetadataError("Frame rate unavailable")

    try:
        rate = Fraction(avg_frame_rate)
    except (ZeroDivisionError, ValueError) as exc:
        raise VideoMetadataError("Invalid frame rate") from exc

    return float(rate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute average duration, frame count, and verify FPS consistency for videos inside a folder."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Folder containing videos (default: DEFAULT_VIDEO_FOLDER in the script)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Include videos within subdirectories",
    )
    args = parser.parse_args()

    root = Path(args.folder).expanduser().resolve() if args.folder else DEFAULT_VIDEO_FOLDER.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Folder not found: {root}", file=sys.stderr)
        return 1

    videos = list(iter_videos(root, args.recursive))
    if not videos:
        print(f"No videos found in {root}")
        return 1

    durations: List[float] = []
    frames: List[float] = []
    fps_values: List[float] = []
    per_video: List[Tuple[Path, float]] = []
    errors: List[Tuple[Path, str]] = []

    for video in videos:
        try:
            duration_s, frame_count, fps = get_video_metadata(video)
        except VideoMetadataError as exc:
            errors.append((video, str(exc)))
            continue

        durations.append(duration_s)
        frames.append(frame_count)
        fps_values.append(fps)
        per_video.append((video, fps))

    analyzed = len(durations)
    if analyzed == 0:
        print("No videos could be analyzed; see errors below:")
        for video, message in errors:
            print(f"- {video}: {message}")
        return 2

    avg_duration = sum(durations) / analyzed
    avg_frames = sum(frames) / analyzed
    avg_fps = sum(fps_values) / analyzed

    skipped = len(errors)
    print(f"Analyzed {analyzed} video(s) in {root} (skipped {skipped}).")
    print(f"Average duration (seconds): {avg_duration:.3f}")
    print(f"Average frame count: {avg_frames:.1f}")
    print(f"Average FPS: {avg_fps:.3f}")

    fps_groups = _group_fps(fps_values)
    if len(fps_groups) == 1:
        print(f"All videos share the same FPS: {fps_groups[0][0]:.6g}")
    else:
        print("FPS mismatch detected. Detailed FPS per video:")
        for video, fps in per_video:
            print(f"- {video}: {fps:.6g} fps")
        print("Distinct FPS values observed:")
        for fps_value, count in fps_groups:
            print(f"- {fps_value:.6g} fps (videos: {count})")

    if errors:
        print("\nVideos skipped:")
        for video, message in errors:
            print(f"- {video}: {message}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VideoMetadataError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
