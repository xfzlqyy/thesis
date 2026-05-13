"""Rig extraction utilities.

Each data type implements a class for easier future extension.
"""

from __future__ import annotations

import abc
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

INSTA_UTILS_DIR = Path(__file__).resolve().parent / "insta_utils"
if str(INSTA_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(INSTA_UTILS_DIR))

try:
    from fs_compat import copy_file_compatible
except ImportError:
    from utils.fs_compat import copy_file_compatible
from insta_extract_frames import extract_insv_frames


class RigExtractor(abc.ABC):
    """Base class for extracting rig images from different data types."""

    @abc.abstractmethod
    def extract(self, input_folder: Path, output_images_dir: Path) -> None:
        """Extract rig images into output_images_dir."""


def load_calibration(input_folder: Path) -> Dict[str, dict]:
    calib_path = input_folder / "info" / "calibration.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with calib_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)
    cams = {}
    for cam in calib.get("cameras", []):
        name = cam.get("name")
        if not name:
            continue
        cams[name] = cam
    if not cams:
        raise ValueError(f"No cameras found in calibration: {calib_path}")
    return cams


class RigExtractor360Raw(RigExtractor):
    """Extract filtered 360 images (left/right) for COLMAP calibration."""

    def __init__(self, filter_ratio: int = 6) -> None:
        self.filter_ratio = max(1, int(filter_ratio))

    def _list_images(self, cam_dir: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png"}
        return sorted(
            p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
        )

    def extract(self, input_folder: Path, output_images_dir: Path) -> None:
        output_images_dir.mkdir(parents=True, exist_ok=True)

        cams = load_calibration(input_folder)
        camera_root = input_folder / "camera"
        if not camera_root.exists():
            raise FileNotFoundError(f"Camera folder not found: {camera_root}")

        image_lists = {}
        for cam_name in cams.keys():
            cam_dir = camera_root / cam_name
            if not cam_dir.exists():
                continue
            image_paths = self._list_images(cam_dir)
            if image_paths:
                image_lists[cam_name] = image_paths

        if not image_lists:
            raise FileNotFoundError(f"No camera images found in {camera_root}")

        lengths = {k: len(v) for k, v in image_lists.items()}
        min_len = min(lengths.values())
        if any(length != min_len for length in lengths.values()):
            print(
                "Warning: camera frame counts differ. "
                f"Using min length {min_len} for synchronized filtering."
            )

        keep_indices = list(range(0, min_len, self.filter_ratio))
        for cam_name, image_paths in image_lists.items():
            out_dir = output_images_dir / cam_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx in keep_indices:
                src = image_paths[idx]
                dst = out_dir / src.name
                copy_file_compatible(src, dst)


class RigExtractor360Images(RigExtractor):
    """Extract perspective views from 360 fisheye images."""

    def __init__(
        self,
        fov_degrees: float = 90.0,
        output_size: Tuple[int, int] = (800, 800),
        stride: int = 1,
    ) -> None:
        self.fov_degrees = fov_degrees
        self.output_size = output_size
        self.stride = max(1, stride)

    def _scaled_intrinsics(
        self,
        cam_info: dict,
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        width = cam_info["width"]
        height = cam_info["height"]
        intr = cam_info["intrinsic"]
        dist = cam_info["distortion"]["params"]

        img_h, img_w = image_shape
        sx = img_w / float(width)
        sy = img_h / float(height)

        k = np.array(
            [
                [intr["fl_x"] * sx, 0.0, intr["cx"] * sx],
                [0.0, intr["fl_y"] * sy, intr["cy"] * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        d = np.array([dist["k1"], dist["k2"], dist["k3"], dist["k4"]], dtype=np.float64)
        return k, d

    def _build_maps(
        self,
        k: np.ndarray,
        d: np.ndarray,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        width, height = self.output_size
        f = width / (2.0 * np.tan(np.radians(self.fov_degrees / 2.0)))
        k_perspective = np.array(
            [
                [f, 0.0, width / 2.0],
                [0.0, f, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        rotations = {
            "front": np.eye(3),
            "left": cv2.Rodrigues(np.array([0.0, np.pi / 4.0 * 0.6, 0.0]))[0],
            "right": cv2.Rodrigues(np.array([0.0, -np.pi / 4.0 * 0.5, 0.0]))[0],
            "top": cv2.Rodrigues(np.array([-np.pi / 4.0, 0.0, 0.0]))[0],
            "bottom": cv2.Rodrigues(np.array([np.pi / 4.0 * 0.5, 0.0, 0.0]))[0],
        }

        maps = {}
        for name, rmat in rotations.items():
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                k, d, rmat, k_perspective, self.output_size, cv2.CV_16SC2
            )
            maps[name] = (map1, map2)
        return maps

    def extract(self, input_folder: Path, output_images_dir: Path) -> None:
        output_images_dir.mkdir(parents=True, exist_ok=True)

        cams = load_calibration(input_folder)
        camera_root = input_folder / "camera"
        if not camera_root.exists():
            raise FileNotFoundError(f"Camera folder not found: {camera_root}")

        for cam_name, cam_info in cams.items():
            cam_dir = camera_root / cam_name
            if not cam_dir.exists():
                continue

            image_paths = sorted(
                p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if not image_paths:
                continue

            first = cv2.imread(str(image_paths[0]))
            if first is None:
                raise RuntimeError(f"Failed to read image: {image_paths[0]}")
            k, d = self._scaled_intrinsics(cam_info, first.shape[:2])
            maps = self._build_maps(k, d)

            for idx, image_path in enumerate(image_paths):
                if idx % self.stride != 0:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                stem = image_path.stem
                for view_name, (map1, map2) in maps.items():
                    view = cv2.remap(
                        image,
                        map1,
                        map2,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    out_name = f"{cam_name}_{stem}_{view_name}.jpg"
                    cv2.imwrite(str(output_images_dir / out_name), view)


class RigExtractorVideo(RigExtractor):
    """Extract frames from videos in the input folder."""

    def __init__(self, fps: int = 1) -> None:
        self.fps = fps

    def _iter_videos(self, input_folder: Path) -> Iterable[Path]:
        exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
        for item in sorted(input_folder.iterdir()):
            if item.is_file() and item.suffix.lower() in exts:
                yield item

    def extract(self, input_folder: Path, output_images_dir: Path) -> None:
        output_images_dir.mkdir(parents=True, exist_ok=True)
        videos = list(self._iter_videos(input_folder))
        if not videos:
            raise FileNotFoundError(f"No video files found in {input_folder}")

        for video_path in videos:
            stem = video_path.stem
            out_pattern = output_images_dir / f"{stem}_%06d.png"
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps={self.fps}",
                str(out_pattern),
            ]
            subprocess.run(cmd, check=True)


class RigExtractorInsta360Video(RigExtractor):
    """Extract full dual-fisheye frames from Insta360 .insv videos."""

    def __init__(
        self,
        fps: int = 1,
        ffmpeg_stream: str = "all",
        sharpness_select: bool = True,
        candidate_multiplier: int = 8,
        sharpness_scale_width: int = 1920,
    ) -> None:
        self.fps = max(1, int(fps))
        stream_mode = ffmpeg_stream.strip().lower()
        if stream_mode not in {"all", "left", "right"}:
            raise ValueError(
                f"Unsupported ffmpeg_stream: {ffmpeg_stream}. Expected all/left/right."
            )
        self.ffmpeg_stream = stream_mode
        self.sharpness_select = bool(sharpness_select)
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.sharpness_scale_width = max(0, int(sharpness_scale_width))

    def _iter_videos(self, input_folder: Path) -> Iterable[Path]:
        exts = {".insv"}
        for item in sorted(input_folder.rglob("*")):
            if item.is_file() and item.suffix.lower() in exts:
                yield item

    def extract(self, input_folder: Path, output_images_dir: Path) -> None:
        output_images_dir.mkdir(parents=True, exist_ok=True)

        videos = list(self._iter_videos(input_folder))
        if not videos:
            raise FileNotFoundError(f"No .insv files found in {input_folder}")
        extract_insv_frames(
            insv_files=videos,
            output_dir=output_images_dir,
            fps=self.fps,
            stream_mode=self.ffmpeg_stream,
            sharpness_select=self.sharpness_select,
            candidate_multiplier=self.candidate_multiplier,
            sharpness_scale_width=self.sharpness_scale_width,
            strict=False,
        )



def get_extractor(data_type: str, **kwargs) -> RigExtractor:
    key = data_type.strip().lower().replace(" ", "").replace("-", "_")
    if key in {"video"}:
        return RigExtractorVideo(fps=kwargs.get("fps", 3))
    if key in {"360images", "360_images", "360"}:
        return RigExtractor360Raw(filter_ratio=kwargs.get("filter_ratio", 6))
    if key in {"insta", "insta360", "insta_360", "insv"}:
        return RigExtractorInsta360Video(
            fps=kwargs.get("fps", 1),
            ffmpeg_stream=kwargs.get("ffmpeg_stream", "all"),
            sharpness_select=kwargs.get("sharpness_select", True),
            candidate_multiplier=kwargs.get("candidate_multiplier", 8),
            sharpness_scale_width=kwargs.get("sharpness_scale_width", 1920),
        )

    raise ValueError(f"Unsupported data_type: {data_type}")
