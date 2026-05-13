"""Known-pose triangulation and bundle adjustment for rigid multi-sensor exports.

This script infers a COLMAP rig/frame structure from an existing sparse model,
extracts and matches features, rewrites the database with the inferred frame
grouping, optionally seeds pose priors for spatial matching,  triangulates
and bundle-adjusts while refining `rig_from_world` and optionally
`sensor_from_rig`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pycolmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.utils.colmap_utils import (  # noqa: E402
    qvec2rotmat,
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    rotmat2qvec,
)
from utils.fs_compat import copy_file_compatible  # noqa: E402


RIG_SYNC_TEXT_FILES = ("cameras.txt", "images.txt", "rigs.txt", "frames.txt")


@dataclass
class RigSensorRecord:
    rig_key: str
    original_id: int
    mapped_id: int
    model_name: str
    width: int
    height: int
    params: np.ndarray
    sensor_from_rig_qvec: np.ndarray
    sensor_from_rig_tvec: np.ndarray
    num_observations: int


@dataclass
class RigImageRecord:
    original_id: int
    name: str
    rig_key: str
    original_camera_id: int
    mapped_camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    center: np.ndarray
    frame_key: str


@dataclass
class RigFrameRecord:
    rig_key: str
    frame_key: str
    mapped_id: int
    image_names: List[str]
    pose_image_name: str


@dataclass
class DatabaseImageBinding:
    image_id: int
    name: str
    camera_id: int
    frame_id: int
    sensor_id: int
    data_id: int


@dataclass
class RigGroupRecord:
    rig_key: str
    mapped_id: int
    ref_sensor_original_id: int
    sensor_original_ids: List[int]


@dataclass
class RigModel:
    rig_dir: Path
    images_dir: Path
    sparse_dir: Path
    grouping_mode: str
    rigs_by_key: Dict[str, RigGroupRecord]
    sensors_by_original_id: Dict[int, RigSensorRecord]
    images_by_name: Dict[str, RigImageRecord]
    frames_by_key: Dict[str, RigFrameRecord]

    @property
    def sensors(self) -> List[RigSensorRecord]:
        return sorted(self.sensors_by_original_id.values(), key=lambda item: item.mapped_id)

    @property
    def rigs(self) -> List[RigGroupRecord]:
        return sorted(self.rigs_by_key.values(), key=lambda item: item.mapped_id)

    @property
    def images(self) -> List[RigImageRecord]:
        return sorted(self.images_by_name.values(), key=lambda item: item.original_id)

    @property
    def frames(self) -> List[RigFrameRecord]:
        return sorted(self.frames_by_key.values(), key=lambda item: item.mapped_id)

    @property
    def camera_id_map(self) -> Dict[int, List[int]]:
        return {
            sensor.original_id: [sensor.mapped_id]
            for sensor in self.sensors
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rig-dir",
        required=True,
        help="Rig root directory containing images/ and sparse/0/.",
    )
    parser.add_argument(
        "--workspace-dir",
        default=None,
        help="Workspace for database and output models. Defaults to <rig-dir>/known_pose_ba.",
    )
    parser.add_argument(
        "--feature-cache-db",
        default=None,
        help=(
            "Path to the reusable feature cache database. Defaults to "
            "<rig-dir>/known_pose_ba_features.db."
        ),
    )
    parser.add_argument(
        "--matcher",
        choices=["auto", "spatial", "vocab", "exhaustive"],
        default="auto",
        help=(
            "Matcher backend. 'auto' prefers spatial matching when pose priors "
            "are available, otherwise falls back to vocab tree / exhaustive."
        ),
    )
    parser.add_argument(
        "--ltg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SuperPoint + LightGlue artifacts instead of COLMAP SIFT features/matches.",
    )
    parser.add_argument(
        "--ltg-hloc-python",
        default=None,
        help="Python executable used for HLOC commands. Default: current Python.",
    )
    parser.add_argument(
        "--ltg-build-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow HLOC feature extraction and matching when <workspace>/ltg artifacts are missing. "
            "Default: fail instead of spending time re-detecting/re-matching."
        ),
    )
    parser.add_argument(
        "--ltg-cpu-threads",
        type=int,
        default=8,
        help="Max CPU threads used by LTG geometric verification. <=0 keeps COLMAP default.",
    )
    parser.add_argument(
        "--ltg-local-window",
        type=int,
        default=4,
        help="Local temporal window used by LTG pair generation.",
    )
    parser.add_argument(
        "--ltg-global-every",
        type=int,
        default=20,
        help="Pick one global anchor every N frames per camera stream for LTG pairing.",
    )
    parser.add_argument(
        "--ltg-netvlad-num-matched",
        type=int,
        default=40,
        help="Additional NetVLAD retrieval pairs merged into LTG pairs. 0 disables retrieval.",
    )
    parser.add_argument(
        "--ltg-spatial-num-neighbors",
        type=int,
        default=20,
        help="Number of spatial KNN neighbors added per image for LTG pairing.",
    )
    parser.add_argument(
        "--ltg-spatial-max-distance",
        type=float,
        default=0.0,
        help="Optional max camera-center distance for LTG spatial pairs. 0 disables the cutoff.",
    )
    parser.add_argument(
        "--vocab-tree-path",
        default=None,
        help="Path to COLMAP vocab tree. Defaults to scripts/utils/vocab.bin when present.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for COLMAP feature extraction/matching and BA when available.",
    )
    parser.add_argument(
        "--refine-focal-length",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow bundle adjustment to refine focal length. Disabled by default.",
    )
    parser.add_argument(
        "--refine-principal-point",
        action="store_true",
        help="Allow bundle adjustment to refine principal point.",
    )
    parser.add_argument(
        "--refine-extra-params",
        action="store_true",
        help="Allow bundle adjustment to refine extra camera parameters.",
    )
    parser.add_argument(
        "--refine-rig-from-world",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow bundle adjustment to refine rig_from_world poses. Enabled by default.",
    )
    parser.add_argument(
        "--refine-sensor-from-rig",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow bundle adjustment to refine sensor_from_rig. Enabled by default.",
    )
    parser.add_argument(
        "--spatial-ignore-z",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore Z when running COLMAP spatial_matcher.",
    )
    parser.add_argument(
        "--spatial-max-num-neighbors",
        type=int,
        default=50,
        help="Maximum number of nearest neighbors to match per image for spatial matching.",
    )
    parser.add_argument(
        "--spatial-min-num-neighbors",
        type=int,
        default=50,
        help="Minimum number of nearest neighbors to match per image for spatial matching.",
    )
    parser.add_argument(
        "--spatial-max-distance",
        type=float,
        default=0.0,
        help=(
            "Maximum Cartesian distance between pose priors for spatial matching. "
            "Set to 0 to rely purely on KNN."
        ),
    )
    parser.add_argument(
        "--share-intrinsics-across-rigs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether different frames of the same rig sensor share intrinsics. "
            "Required for rigid multi-sensor BA and enabled by default."
        ),
    )
    parser.add_argument(
        "--frame-center-tolerance",
        type=float,
        default=1e-6,
        help=(
            "Maximum camera-center distance for grouping images into the same "
            "multi-sensor frame when inferring the rig structure."
        ),
    )
    parser.add_argument(
        "--frame-grouping",
        choices=["auto", "name", "center"],
        default="auto",
        help=(
            "How to infer rig frames. 'name' groups images by <rig>_<view>_<frame> "
            "filename pattern, 'center' groups by camera-center proximity, and "
            "'auto' prefers filename grouping when possible."
        ),
    )
    return parser.parse_args()


def run_command(cmd: List[str], env: Dict[str, str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def ensure_clean_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def read_sparse_model(sparse_dir: Path):
    cameras_txt = sparse_dir / "cameras.txt"
    cameras_bin = sparse_dir / "cameras.bin"
    images_txt = sparse_dir / "images.txt"
    images_bin = sparse_dir / "images.bin"

    if cameras_txt.exists():
        cameras = read_cameras_text(str(cameras_txt))
    elif cameras_bin.exists():
        cameras = read_cameras_binary(str(cameras_bin))
    else:
        raise FileNotFoundError(f"No cameras.txt/cameras.bin found in {sparse_dir}")

    if images_txt.exists():
        images = read_images_text(str(images_txt))
    elif images_bin.exists():
        images = read_images_binary(str(images_bin))
    else:
        raise FileNotFoundError(f"No images.txt/images.bin found in {sparse_dir}")

    return cameras, images


def camera_center_from_colmap_pose(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rmat = qvec2rotmat(qvec)
    return -rmat.T @ np.asarray(tvec, dtype=np.float64)


def average_qvec(qvecs: List[np.ndarray]) -> np.ndarray:
    aligned = np.stack([np.asarray(qvec, dtype=np.float64).copy() for qvec in qvecs], axis=0)
    ref = aligned[0]
    for idx in range(1, aligned.shape[0]):
        if np.dot(aligned[idx], ref) < 0.0:
            aligned[idx] *= -1.0
    mean_qvec = aligned.mean(axis=0)
    mean_qvec /= np.linalg.norm(mean_qvec)
    return mean_qvec


def pose_inverse(qvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rmat = qvec2rotmat(qvec)
    rmat_inv = rmat.T
    qvec_inv = rotmat2qvec(rmat_inv)
    tvec_inv = -rmat_inv @ np.asarray(tvec, dtype=np.float64)
    return qvec_inv, tvec_inv


def compose_poses(
    qvec_ab: np.ndarray,
    tvec_ab: np.ndarray,
    qvec_bc: np.ndarray,
    tvec_bc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rmat_ab = qvec2rotmat(qvec_ab)
    rmat_bc = qvec2rotmat(qvec_bc)
    rmat_ac = rmat_ab @ rmat_bc
    tvec_ac = rmat_ab @ np.asarray(tvec_bc, dtype=np.float64) + np.asarray(tvec_ab, dtype=np.float64)
    return rotmat2qvec(rmat_ac), tvec_ac


def relative_pose(
    qvec_target: np.ndarray,
    tvec_target: np.ndarray,
    qvec_source: np.ndarray,
    tvec_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qvec_source_inv, tvec_source_inv = pose_inverse(qvec_source, tvec_source)
    return compose_poses(qvec_target, tvec_target, qvec_source_inv, tvec_source_inv)


def infer_frame_groups_by_name(
    sparse_images: Dict[int, object],
) -> tuple[Dict[str, List[int]], Dict[int, str], Dict[str, str]]:
    grouped_original_ids: Dict[str, List[int]] = defaultdict(list)
    image_to_frame_key: Dict[int, str] = {}
    frame_to_rig_key: Dict[str, str] = {}

    for original_id, image in sorted(sparse_images.items()):
        stem = Path(image.name).stem
        parts = stem.split("_", 2)
        if len(parts) < 3:
            raise ValueError(
                f"Image name does not match <rig>_<view>_<frame> pattern: {image.name}"
            )
        rig_key = parts[0]
        frame_token = parts[2]
        if not rig_key or not frame_token:
            raise ValueError(
                f"Image name does not match <rig>_<view>_<frame> pattern: {image.name}"
            )

        frame_key = f"{rig_key}:{frame_token}"
        grouped_original_ids[frame_key].append(int(original_id))
        image_to_frame_key[int(original_id)] = frame_key
        frame_to_rig_key[frame_key] = rig_key

    return dict(grouped_original_ids), image_to_frame_key, frame_to_rig_key


def infer_frame_groups_by_center(
    sparse_images: Dict[int, object],
    center_tolerance: float,
) -> tuple[Dict[str, List[int]], Dict[int, str], Dict[str, str]]:
    if center_tolerance <= 0:
        raise ValueError("--frame-center-tolerance must be positive")

    grouped_original_ids: Dict[str, List[int]] = {}
    image_to_frame_key: Dict[int, str] = {}
    frame_to_rig_key: Dict[str, str] = {}
    frame_centers: List[np.ndarray] = []
    frame_keys: List[str] = []

    for original_id, image in sorted(sparse_images.items()):
        center = camera_center_from_colmap_pose(
            np.asarray(image.qvec, dtype=np.float64),
            np.asarray(image.tvec, dtype=np.float64),
        )
        matched_index = None
        for idx, ref_center in enumerate(frame_centers):
            if np.linalg.norm(center - ref_center) <= center_tolerance:
                matched_index = idx
                break
        if matched_index is None:
            matched_index = len(frame_centers)
            frame_centers.append(center)
            frame_keys.append(f"frame_{matched_index:06d}")
            grouped_original_ids[frame_keys[-1]] = []
            frame_to_rig_key[frame_keys[-1]] = "rig_000001"
        frame_key = frame_keys[matched_index]
        grouped_original_ids[frame_key].append(int(original_id))
        image_to_frame_key[int(original_id)] = frame_key

    return grouped_original_ids, image_to_frame_key, frame_to_rig_key


def infer_frame_groups(
    sparse_images: Dict[int, object],
    center_tolerance: float,
    grouping_mode: str,
) -> tuple[Dict[str, List[int]], Dict[int, str], Dict[str, str], str]:
    if grouping_mode == "name":
        grouped_original_ids, image_to_frame_key, frame_to_rig_key = (
            infer_frame_groups_by_name(sparse_images)
        )
        return grouped_original_ids, image_to_frame_key, frame_to_rig_key, "name"

    if grouping_mode == "center":
        grouped_original_ids, image_to_frame_key, frame_to_rig_key = (
            infer_frame_groups_by_center(sparse_images, center_tolerance)
        )
        return grouped_original_ids, image_to_frame_key, frame_to_rig_key, "center"

    if grouping_mode != "auto":
        raise ValueError(f"Unsupported --frame-grouping mode: {grouping_mode}")

    try:
        grouped_original_ids, image_to_frame_key, frame_to_rig_key = (
            infer_frame_groups_by_name(sparse_images)
        )
        return grouped_original_ids, image_to_frame_key, frame_to_rig_key, "name"
    except ValueError:
        grouped_original_ids, image_to_frame_key, frame_to_rig_key = (
            infer_frame_groups_by_center(sparse_images, center_tolerance)
        )
        return grouped_original_ids, image_to_frame_key, frame_to_rig_key, "center"


def load_rig_model(
    rig_dir: Path,
    share_intrinsics_across_rigs: bool = True,
    frame_center_tolerance: float = 1e-6,
    frame_grouping: str = "auto",
) -> RigModel:
    if not share_intrinsics_across_rigs:
        raise ValueError(
            "rig-aware known_pose_ba requires shared intrinsics per sensor. "
            "Use the default --share-intrinsics-across-rigs."
        )

    images_dir = rig_dir / "images"
    sparse_dir = rig_dir / "sparse" / "0"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not sparse_dir.exists():
        raise FileNotFoundError(f"Sparse model directory not found: {sparse_dir}")

    sparse_cameras, sparse_images = read_sparse_model(sparse_dir)
    if not sparse_cameras:
        raise RuntimeError(f"No cameras found in {sparse_dir}")
    if not sparse_images:
        raise RuntimeError(f"No images found in {sparse_dir}")

    grouped_original_ids, image_to_frame_key, frame_to_rig_key, grouping_mode = (
        infer_frame_groups(
        sparse_images=sparse_images,
        center_tolerance=frame_center_tolerance,
        grouping_mode=frame_grouping,
        )
    )

    images_by_name: Dict[str, RigImageRecord] = {}
    sensor_observation_counts: Dict[int, int] = defaultdict(int)
    sensor_rig_keys: Dict[int, str] = {}
    rig_sensor_observation_counts: Dict[str, Dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for original_id, image in sorted(sparse_images.items()):
        image_path = images_dir / image.name
        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in sparse model is missing: {image_path}")
        if image.name in images_by_name:
            raise ValueError(f"Duplicate image name found in sparse model: {image.name}")

        original_camera_id = int(image.camera_id)
        if original_camera_id not in sparse_cameras:
            raise KeyError(f"Image {image.name} references missing camera_id={image.camera_id}")

        qvec = np.asarray(image.qvec, dtype=np.float64)
        tvec = np.asarray(image.tvec, dtype=np.float64)
        frame_key = image_to_frame_key[int(original_id)]
        rig_key = frame_to_rig_key[frame_key]
        center = camera_center_from_colmap_pose(qvec, tvec)

        if original_camera_id in sensor_rig_keys and sensor_rig_keys[original_camera_id] != rig_key:
            raise RuntimeError(
                f"Camera {original_camera_id} appears in multiple rig groups: "
                f"{sensor_rig_keys[original_camera_id]} vs {rig_key}"
            )
        sensor_rig_keys[original_camera_id] = rig_key
        sensor_observation_counts[original_camera_id] += 1
        rig_sensor_observation_counts[rig_key][original_camera_id] += 1
        images_by_name[image.name] = RigImageRecord(
            original_id=int(original_id),
            name=image.name,
            rig_key=rig_key,
            original_camera_id=original_camera_id,
            mapped_camera_id=-1,
            qvec=qvec,
            tvec=tvec,
            center=center,
            frame_key=frame_key,
        )

    if not sensor_observation_counts:
        raise RuntimeError("Failed to infer any rig sensors from the sparse model")

    rig_ref_sensor_original_ids: Dict[str, int] = {}
    for rig_key, counts in rig_sensor_observation_counts.items():
        ref_sensor_original_id = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        rig_ref_sensor_original_ids[rig_key] = int(ref_sensor_original_id)

    grouped_image_names: Dict[str, List[str]] = {}
    for frame_key, original_ids in grouped_original_ids.items():
        rig_key = frame_to_rig_key[frame_key]
        names: List[str] = []
        seen_sensor_ids = set()
        for original_id in sorted(original_ids):
            image = sparse_images[original_id]
            sensor_id = int(image.camera_id)
            if sensor_id in seen_sensor_ids:
                raise ValueError(
                    f"Frame {frame_key} contains multiple images for sensor {sensor_id}. "
                    "Each frame must contain at most one image per sensor."
                )
            seen_sensor_ids.add(sensor_id)
            if sensor_rig_keys[sensor_id] != rig_key:
                raise RuntimeError(
                    f"Frame {frame_key} mixes sensors from different rig groups: "
                    f"camera {sensor_id} belongs to {sensor_rig_keys[sensor_id]}, not {rig_key}"
                )
            names.append(image.name)
        grouped_image_names[frame_key] = names

    sensor_relative_qvecs: Dict[int, List[np.ndarray]] = defaultdict(list)
    sensor_relative_tvecs: Dict[int, List[np.ndarray]] = defaultdict(list)
    for ref_sensor_original_id in rig_ref_sensor_original_ids.values():
        sensor_relative_qvecs[ref_sensor_original_id].append(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        sensor_relative_tvecs[ref_sensor_original_id].append(np.zeros(3, dtype=np.float64))

    for frame_key, image_names in grouped_image_names.items():
        rig_key = frame_to_rig_key[frame_key]
        ref_sensor_original_id = rig_ref_sensor_original_ids[rig_key]
        ref_image_name = next(
            (name for name in image_names if images_by_name[name].original_camera_id == ref_sensor_original_id),
            None,
        )
        if ref_image_name is None:
            continue

        ref_image = images_by_name[ref_image_name]
        for image_name in image_names:
            rig_image = images_by_name[image_name]
            qvec_rel, tvec_rel = relative_pose(
                rig_image.qvec,
                rig_image.tvec,
                ref_image.qvec,
                ref_image.tvec,
            )
            sensor_relative_qvecs[rig_image.original_camera_id].append(qvec_rel)
            sensor_relative_tvecs[rig_image.original_camera_id].append(tvec_rel)

    rig_groups_by_key: Dict[str, RigGroupRecord] = {}
    for rig_idx, rig_key in enumerate(sorted(rig_sensor_observation_counts.keys()), start=1):
        rig_groups_by_key[rig_key] = RigGroupRecord(
            rig_key=rig_key,
            mapped_id=rig_idx,
            ref_sensor_original_id=int(rig_ref_sensor_original_ids[rig_key]),
            sensor_original_ids=sorted(rig_sensor_observation_counts[rig_key].keys()),
        )

    sensors_by_original_id: Dict[int, RigSensorRecord] = {}
    next_camera_id = 1
    for original_camera_id, source_camera in sorted(sparse_cameras.items()):
        rig_key = sensor_rig_keys.get(int(original_camera_id))
        if rig_key is None:
            raise RuntimeError(
                f"Failed to infer rig group for sensor {original_camera_id}"
            )
        if original_camera_id not in sensor_relative_qvecs:
            ref_sensor_original_id = rig_ref_sensor_original_ids[rig_key]
            raise RuntimeError(
                f"Sensor {original_camera_id} never co-occurred with the reference sensor "
                f"{ref_sensor_original_id} in rig '{rig_key}'; cannot infer a fixed "
                "sensor_from_rig transform."
            )

        mean_qvec = average_qvec(sensor_relative_qvecs[original_camera_id])
        mean_tvec = np.mean(np.stack(sensor_relative_tvecs[original_camera_id], axis=0), axis=0)
        mapped_camera_id = next_camera_id
        next_camera_id += 1

        sensors_by_original_id[int(original_camera_id)] = RigSensorRecord(
            rig_key=rig_key,
            original_id=int(original_camera_id),
            mapped_id=mapped_camera_id,
            model_name=str(source_camera.model),
            width=int(source_camera.width),
            height=int(source_camera.height),
            params=np.asarray(source_camera.params, dtype=np.float64),
            sensor_from_rig_qvec=mean_qvec,
            sensor_from_rig_tvec=np.asarray(mean_tvec, dtype=np.float64),
            num_observations=int(sensor_observation_counts[original_camera_id]),
        )

    for rig_image in images_by_name.values():
        rig_image.mapped_camera_id = sensors_by_original_id[rig_image.original_camera_id].mapped_id

    frames_by_key: Dict[str, RigFrameRecord] = {}
    next_frame_id = 1
    for frame_key, image_names in sorted(grouped_image_names.items(), key=lambda item: item[0]):
        rig_key = frame_to_rig_key[frame_key]
        ref_sensor_original_id = rig_ref_sensor_original_ids[rig_key]
        pose_image_name = next(
            (name for name in image_names if images_by_name[name].original_camera_id == ref_sensor_original_id),
            image_names[0],
        )
        frames_by_key[frame_key] = RigFrameRecord(
            rig_key=rig_key,
            frame_key=frame_key,
            mapped_id=next_frame_id,
            image_names=sorted(image_names, key=lambda name: images_by_name[name].mapped_camera_id),
            pose_image_name=pose_image_name,
        )
        next_frame_id += 1

    return RigModel(
        rig_dir=rig_dir,
        images_dir=images_dir,
        sparse_dir=sparse_dir,
        grouping_mode=grouping_mode,
        rigs_by_key=rig_groups_by_key,
        sensors_by_original_id=sensors_by_original_id,
        images_by_name=images_by_name,
        frames_by_key=frames_by_key,
    )


def camera_model_id(model_name: str) -> int:
    try:
        return int(getattr(pycolmap.CameraModelId, model_name))
    except AttributeError as exc:
        raise ValueError(f"Unsupported camera model for pycolmap: {model_name}") from exc


def camera_params_blob(params: np.ndarray) -> sqlite3.Binary:
    return sqlite3.Binary(np.asarray(params, dtype=np.float64).tobytes())


def bool_to_colmap_flag(value: bool) -> str:
    return "1" if value else "0"


def float64_blob(array: np.ndarray) -> sqlite3.Binary:
    return sqlite3.Binary(np.asarray(array, dtype=np.float64).tobytes())


def rigid3d_blob(qvec: np.ndarray, tvec: np.ndarray) -> sqlite3.Binary:
    qvec = np.asarray(qvec, dtype=np.float64)
    tvec = np.asarray(tvec, dtype=np.float64)
    if qvec.shape != (4,):
        raise ValueError(f"Expected qvec shape (4,), got {qvec.shape}")
    if tvec.shape != (3,):
        raise ValueError(f"Expected tvec shape (3,), got {tvec.shape}")
    return float64_blob(np.concatenate([qvec, tvec], axis=0))


def write_image_list(rig_model: RigModel, image_list_path: Path) -> None:
    names = [record.name for record in rig_model.images]
    image_list_path.write_text("\n".join(names) + "\n", encoding="utf-8")


def default_feature_cache_db_path(rig_dir: Path) -> Path:
    return rig_dir / "known_pose_ba_features.db"


def database_has_usable_features(
    database_path: Path,
    expected_image_names: List[str],
) -> bool:
    if not database_path.exists():
        return False

    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        table_names = {
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        required_tables = {"images", "keypoints", "descriptors"}
        if not required_tables.issubset(table_names):
            return False

        image_rows = cur.execute("SELECT name FROM images ORDER BY name").fetchall()
        db_image_names = [row[0] for row in image_rows]
        if db_image_names != sorted(expected_image_names):
            return False

        num_images = len(expected_image_names)
        num_keypoints = int(cur.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0])
        num_descriptors = int(cur.execute("SELECT COUNT(*) FROM descriptors").fetchone()[0])
        if num_keypoints != num_images or num_descriptors != num_images:
            return False

        return True
    finally:
        conn.close()


def reset_database_for_run(database_path: Path) -> None:
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        table_names = {
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table_name in [
            "matches",
            "two_view_geometries",
            "pose_priors",
            "frame_data",
            "frames",
            "rig_sensors",
            "rigs",
        ]:
            if table_name in table_names:
                cur.execute(f"DELETE FROM {table_name}")

        cur.execute("DROP TABLE IF EXISTS pose_priors")
        cur.execute(
            """
            CREATE TABLE pose_priors
               (image_id                   INTEGER  PRIMARY KEY  NOT NULL,
                position                   BLOB,
                coordinate_system          INTEGER               NOT NULL,
                position_covariance        BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
            """
        )
        conn.commit()
    finally:
        conn.close()


def run_feature_extractor(
    rig_model: RigModel,
    database_path: Path,
    use_gpu: bool,
    image_list_path: Path,
    env: Dict[str, str],
) -> Path:
    seed_camera = rig_model.sensors[0]
    camera_params = ",".join(str(value) for value in seed_camera.params.tolist())
    gpu_flag = "1" if use_gpu else "0"
    uses_sensor_folders = any(Path(record.name).parent != Path(".") for record in rig_model.images)

    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()

    run_command(
        ["colmap", "database_creator", "--database_path", str(database_path)],
        env=env,
    )

    feature_cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(rig_model.images_dir),
        "--image_list_path",
        str(image_list_path),
        "--FeatureExtraction.use_gpu",
        gpu_flag,
        "--ImageReader.camera_model",
        seed_camera.model_name,
        "--ImageReader.camera_params",
        camera_params,
    ]
    if uses_sensor_folders:
        feature_cmd += ["--ImageReader.single_camera_per_folder", "1"]
    else:
        feature_cmd += ["--ImageReader.single_camera", "0"]
    run_command(feature_cmd, env=env)
    return database_path


def copy_database_compatible(src: Path, dst: Path) -> None:
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        # exFAT and similar filesystems can reject chmod/copystat even though the file is writable.
        shutil.copyfile(src, dst)


def create_database(
    rig_model: RigModel,
    workspace_dir: Path,
    use_gpu: bool,
    image_list_path: Path,
    env: Dict[str, str],
    feature_cache_db_path: Path,
) -> Path:
    database_path = workspace_dir / "database.db"
    expected_image_names = [record.name for record in rig_model.images]
    workspace_dir.mkdir(parents=True, exist_ok=True)

    if database_has_usable_features(database_path, expected_image_names):
        print(f"Reusing workspace features: {database_path}", flush=True)
        reset_database_for_run(database_path)
        return database_path

    if feature_cache_db_path != database_path and database_has_usable_features(
        feature_cache_db_path,
        expected_image_names,
    ):
        print(f"Reusing feature cache: {feature_cache_db_path}", flush=True)
        copy_database_compatible(feature_cache_db_path, database_path)
        reset_database_for_run(database_path)
        return database_path

    build_database_path = (
        feature_cache_db_path if feature_cache_db_path != database_path else database_path
    )
    print(f"Building feature database: {build_database_path}", flush=True)
    run_feature_extractor(
        rig_model=rig_model,
        database_path=build_database_path,
        use_gpu=use_gpu,
        image_list_path=image_list_path,
        env=env,
    )

    if build_database_path != database_path:
        copy_database_compatible(build_database_path, database_path)

    reset_database_for_run(database_path)
    return database_path


def import_images_for_ltg_database(rig_model: RigModel, database_path: Path) -> None:
    seed_camera = rig_model.sensors[0]
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = OFF")
        for table_name in [
            "two_view_geometries",
            "matches",
            "descriptors",
            "keypoints",
            "pose_priors",
            "frame_data",
            "frames",
            "rig_sensors",
            "rigs",
            "images",
            "cameras",
        ]:
            cur.execute(f"DELETE FROM {table_name}")

        cur.execute(
            """
            INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                seed_camera.mapped_id,
                camera_model_id(seed_camera.model_name),
                seed_camera.width,
                seed_camera.height,
                camera_params_blob(seed_camera.params),
                1,
            ),
        )
        for image_id, record in enumerate(rig_model.images, start=1):
            cur.execute(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
                (int(image_id), record.name, seed_camera.mapped_id),
            )
        cur.execute(
            "DELETE FROM sqlite_sequence WHERE name IN ('cameras', 'images', 'frames', 'rigs')"
        )
        conn.commit()
    finally:
        conn.close()


def run_ltg_matching(
    args: argparse.Namespace,
    rig_model: RigModel,
    database_path: Path,
    workspace_dir: Path,
    env: Dict[str, str],
) -> object:
    ltg_utils_dir = Path(__file__).resolve().parent / "metacam_utils"
    if str(ltg_utils_dir) not in sys.path:
        sys.path.insert(0, str(ltg_utils_dir))
    from ltg_utils import (
        LtgPairingConfig,
        build_spatial_entries_from_rig_images,
        find_existing_ltg_artifacts,
        import_ltg_into_database,
        run_hloc_feature_pipeline,
    )

    entries = build_spatial_entries_from_rig_images(rig_model.images)
    pairing = LtgPairingConfig(
        local_window=int(args.ltg_local_window),
        global_every=int(args.ltg_global_every),
        netvlad_num_matched=int(args.ltg_netvlad_num_matched),
        spatial_num_neighbors=int(args.ltg_spatial_num_neighbors),
        spatial_max_distance=float(args.ltg_spatial_max_distance),
    )
    artifacts = find_existing_ltg_artifacts(
        workspace_dir / "ltg",
        expected_image_names=(entry.name for entry in entries),
        expected_pairing=pairing,
    )
    if artifacts is None:
        if not bool(args.ltg_build_missing):
            raise FileNotFoundError(
                "LTG artifacts are missing or incompatible under "
                f"{workspace_dir / 'ltg'}. Refusing to re-detect features/re-match by default. "
                "Reuse an existing LTG directory or pass --ltg-build-missing to generate it."
            )
        artifacts = run_hloc_feature_pipeline(
            image_dir=rig_model.images_dir,
            work_dir=workspace_dir / "ltg",
            entries=entries,
            pairing=pairing,
            hloc_python=args.ltg_hloc_python,
            env=env,
        )
    else:
        print(f"Reusing existing LTG artifacts from {artifacts.work_dir}", flush=True)
    import_ltg_into_database(
        database_path=database_path,
        artifacts=artifacts,
        verbose=True,
        cpu_threads=int(args.ltg_cpu_threads),
        hloc_python=args.ltg_hloc_python,
        env=env,
    )
    return artifacts


def patch_database_with_rig_model(database_path: Path, rig_model: RigModel) -> None:
    name_to_image = rig_model.images_by_name
    sensors = rig_model.sensors

    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = OFF")
        db_images = cur.execute(
            "SELECT image_id, name FROM images ORDER BY image_id"
        ).fetchall()
        db_names = {name for _, name in db_images}
        model_names = set(name_to_image.keys())
        if db_names != model_names:
            missing = sorted(model_names - db_names)
            extra = sorted(db_names - model_names)
            raise RuntimeError(
                f"Database/image model mismatch. Missing in database: {missing[:5]}, "
                f"extra in database: {extra[:5]}"
            )

        name_to_db_image_id = {name: int(image_id) for image_id, name in db_images}

        cur.execute("DELETE FROM cameras")
        for sensor in sensors:
            cur.execute(
                """
                INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sensor.mapped_id,
                    camera_model_id(sensor.model_name),
                    sensor.width,
                    sensor.height,
                    camera_params_blob(sensor.params),
                    1,
                ),
            )

        cur.execute("DELETE FROM rig_sensors")
        cur.execute("DELETE FROM frame_data")
        cur.execute("DELETE FROM frames")
        cur.execute("DELETE FROM rigs")
        for rig_group in rig_model.rigs:
            ref_sensor = rig_model.sensors_by_original_id[rig_group.ref_sensor_original_id]
            cur.execute(
                "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) VALUES (?, ?, ?)",
                (rig_group.mapped_id, ref_sensor.mapped_id, int(pycolmap.SensorType.CAMERA)),
            )
            for original_sensor_id in rig_group.sensor_original_ids:
                sensor = rig_model.sensors_by_original_id[original_sensor_id]
                if sensor.mapped_id == ref_sensor.mapped_id:
                    continue
                cur.execute(
                    """
                    INSERT INTO rig_sensors(rig_id, sensor_id, sensor_type, sensor_from_rig)
                    VALUES (?, ?, ?, ?)
                    """,
                    (rig_group.mapped_id, sensor.mapped_id, int(pycolmap.SensorType.CAMERA), None),
                )

        for name, rig_image in name_to_image.items():
            cur.execute(
                "UPDATE images SET camera_id = ? WHERE image_id = ?",
                (rig_image.mapped_camera_id, name_to_db_image_id[name]),
            )

        for frame in rig_model.frames:
            rig_group = rig_model.rigs_by_key[frame.rig_key]
            cur.execute(
                "INSERT INTO frames(frame_id, rig_id) VALUES (?, ?)",
                (frame.mapped_id, rig_group.mapped_id),
            )
            for image_name in frame.image_names:
                rig_image = name_to_image[image_name]
                cur.execute(
                    """
                    INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        frame.mapped_id,
                        name_to_db_image_id[image_name],
                        rig_image.mapped_camera_id,
                        int(pycolmap.SensorType.CAMERA),
                    ),
                )

        conn.commit()
    finally:
        conn.close()


def populate_database_sensor_from_rig(database_path: Path, rig_model: RigModel) -> None:
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        cur.execute("UPDATE rig_sensors SET sensor_from_rig = NULL")

        for rig_group in rig_model.rigs:
            ref_sensor = rig_model.sensors_by_original_id[rig_group.ref_sensor_original_id]
            for original_sensor_id in rig_group.sensor_original_ids:
                sensor = rig_model.sensors_by_original_id[original_sensor_id]
                if sensor.mapped_id == ref_sensor.mapped_id:
                    continue
                cur.execute(
                    """
                    UPDATE rig_sensors
                    SET sensor_from_rig = ?
                    WHERE rig_id = ? AND sensor_id = ? AND sensor_type = ?
                    """,
                    (
                        rigid3d_blob(
                            sensor.sensor_from_rig_qvec,
                            sensor.sensor_from_rig_tvec,
                        ),
                        rig_group.mapped_id,
                        sensor.mapped_id,
                        int(pycolmap.SensorType.CAMERA),
                    ),
                )
                if cur.rowcount != 1:
                    raise RuntimeError(
                        "Failed to update sensor_from_rig for "
                        f"rig_id={rig_group.mapped_id}, sensor_id={sensor.mapped_id}"
                    )

        conn.commit()
    finally:
        conn.close()


def write_pose_priors(database_path: Path, rig_model: RigModel) -> int:
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        columns = [row[1] for row in cur.execute("PRAGMA table_info(pose_priors)").fetchall()]
        if not columns:
            raise RuntimeError("pose_priors table is missing from the database")

        cur.execute("DELETE FROM pose_priors")

        nan_covariance_blob = float64_blob(np.full((3, 3), np.nan, dtype=np.float64))
        nan_gravity_blob = float64_blob(np.full(3, np.nan, dtype=np.float64))
        coordinate_system = int(pycolmap.PosePriorCoordinateSystem.CARTESIAN)

        if "image_id" in columns:
            rows = cur.execute(
                "SELECT image_id, name FROM images ORDER BY image_id"
            ).fetchall()
            for image_id, image_name in rows:
                if image_name not in rig_model.images_by_name:
                    raise KeyError(f"Image {image_name} not found in inferred rig model")
                rig_image = rig_model.images_by_name[image_name]
                cur.execute(
                    """
                    INSERT INTO pose_priors(image_id, position, coordinate_system, position_covariance)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        int(image_id),
                        float64_blob(rig_image.center),
                        coordinate_system,
                        nan_covariance_blob,
                    ),
                )
        elif {"corr_data_id", "corr_sensor_id", "corr_sensor_type"}.issubset(columns):
            rows = cur.execute(
                """
                SELECT
                    images.name,
                    frame_data.data_id,
                    frame_data.sensor_id,
                    frame_data.sensor_type
                FROM images
                JOIN frame_data ON frame_data.data_id = images.image_id
                ORDER BY images.image_id
                """
            ).fetchall()
            for image_name, corr_data_id, corr_sensor_id, corr_sensor_type in rows:
                if image_name not in rig_model.images_by_name:
                    raise KeyError(f"Image {image_name} not found in inferred rig model")
                rig_image = rig_model.images_by_name[image_name]
                cur.execute(
                    """
                    INSERT INTO pose_priors(
                        corr_data_id,
                        corr_sensor_id,
                        corr_sensor_type,
                        position,
                        position_covariance,
                        gravity,
                        coordinate_system
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(corr_data_id),
                        int(corr_sensor_id),
                        int(corr_sensor_type),
                        float64_blob(rig_image.center),
                        nan_covariance_blob,
                        nan_gravity_blob,
                        coordinate_system,
                    ),
                )
        else:
            raise RuntimeError(
                f"Unsupported pose_priors schema columns: {columns}"
            )

        conn.commit()
        num_pose_priors = int(cur.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0])
    finally:
        conn.close()

    expected_num_priors = len(rig_model.images_by_name)
    if num_pose_priors != expected_num_priors:
        raise RuntimeError(
            f"Expected {expected_num_priors} pose priors, but wrote {num_pose_priors}"
        )
    return num_pose_priors


def run_matcher(
    database_path: Path,
    matcher: str,
    vocab_tree_path: Path | None,
    use_gpu: bool,
    spatial_ignore_z: bool,
    spatial_max_num_neighbors: int,
    spatial_min_num_neighbors: int,
    spatial_max_distance: float,
    env: Dict[str, str],
) -> str:
    if spatial_max_num_neighbors <= 0:
        raise ValueError("--spatial-max-num-neighbors must be positive")
    if spatial_min_num_neighbors < 0:
        raise ValueError("--spatial-min-num-neighbors must be non-negative")
    if spatial_min_num_neighbors > spatial_max_num_neighbors:
        raise ValueError(
            "--spatial-min-num-neighbors cannot exceed --spatial-max-num-neighbors"
        )
    if spatial_max_distance < 0:
        raise ValueError("--spatial-max-distance must be non-negative")

    gpu_flag = bool_to_colmap_flag(use_gpu)
    chosen = matcher
    if chosen == "auto":
        chosen = "spatial"

    if chosen == "spatial":
        run_command(
            [
                "colmap",
                "spatial_matcher",
                "--database_path",
                str(database_path),
                "--FeatureMatching.rig_verification",
                "0",
                "--FeatureMatching.use_gpu",
                gpu_flag,
                "--SpatialMatching.ignore_z",
                bool_to_colmap_flag(spatial_ignore_z),
                "--SpatialMatching.max_num_neighbors",
                str(spatial_max_num_neighbors),
                "--SpatialMatching.min_num_neighbors",
                str(spatial_min_num_neighbors),
                "--SpatialMatching.max_distance",
                str(spatial_max_distance),
            ],
            env=env,
        )
    elif chosen == "vocab":
        if vocab_tree_path is None or not vocab_tree_path.exists():
            raise FileNotFoundError(
                f"Vocab tree matcher requested but file is missing: {vocab_tree_path}"
            )
        run_command(
            [
                "colmap",
                "vocab_tree_matcher",
                "--database_path",
                str(database_path),
                "--VocabTreeMatching.vocab_tree_path",
                str(vocab_tree_path),
                "--FeatureMatching.rig_verification",
                "0",
                "--FeatureMatching.use_gpu",
                gpu_flag,
            ],
            env=env,
        )
    else:
        run_command(
            [
                "colmap",
                "exhaustive_matcher",
                "--database_path",
                str(database_path),
                "--FeatureMatching.rig_verification",
                "0",
                "--FeatureMatching.use_gpu",
                gpu_flag,
            ],
            env=env,
        )
    return chosen


def summarize_database_matching(database_path: Path) -> Dict[str, int]:
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        table_names = {
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        def count_rows(table_name: str) -> int:
            if table_name not in table_names:
                return 0
            return int(cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])

        def sum_rows_column(table_name: str) -> int:
            if table_name not in table_names:
                return 0
            columns = {
                row[1]
                for row in cur.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            if "rows" not in columns:
                return count_rows(table_name)
            value = cur.execute(f"SELECT COALESCE(SUM(rows), 0) FROM {table_name}").fetchone()[0]
            return int(value or 0)

        return {
            "num_images": count_rows("images"),
            "num_pose_priors": count_rows("pose_priors"),
            "num_matched_image_pairs": count_rows("matches"),
            "num_verified_image_pairs": count_rows("two_view_geometries"),
            "num_matches": sum_rows_column("matches"),
            "num_inlier_matches": sum_rows_column("two_view_geometries"),
        }
    finally:
        conn.close()


def load_database_bindings(database_path: Path) -> List[DatabaseImageBinding]:
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT
                images.image_id,
                images.name,
                images.camera_id,
                frame_data.frame_id,
                frame_data.sensor_id,
                frame_data.data_id
            FROM images
            JOIN frame_data ON frame_data.data_id = images.image_id
            ORDER BY images.image_id
            """
        ).fetchall()
    finally:
        conn.close()

    bindings = [
        DatabaseImageBinding(
            image_id=int(image_id),
            name=name,
            camera_id=int(camera_id),
            frame_id=int(frame_id),
            sensor_id=int(sensor_id),
            data_id=int(data_id),
        )
        for image_id, name, camera_id, frame_id, sensor_id, data_id in rows
    ]
    if not bindings:
        raise RuntimeError(f"No image/frame bindings found in {database_path}")
    return bindings


def colmap_pose_to_rigid3d(qvec: np.ndarray, tvec: np.ndarray) -> pycolmap.Rigid3d:
    qw, qx, qy, qz = [float(value) for value in qvec.tolist()]
    return pycolmap.Rigid3d(
        {
            "rotation": {"quat": np.array([qx, qy, qz, qw], dtype=np.float64)},
            "translation": np.asarray(tvec, dtype=np.float64),
        }
    )


def build_known_pose_reconstruction(
    rig_model: RigModel,
    bindings: Iterable[DatabaseImageBinding],
) -> pycolmap.Reconstruction:
    reconstruction = pycolmap.Reconstruction()
    for sensor in rig_model.sensors:
        reconstruction.add_camera(
            pycolmap.Camera(
                {
                    "camera_id": sensor.mapped_id,
                    "model": getattr(pycolmap.CameraModelId, sensor.model_name),
                    "width": sensor.width,
                    "height": sensor.height,
                    "params": sensor.params.tolist(),
                }
            )
        )

    for rig_group in rig_model.rigs:
        ref_sensor = rig_model.sensors_by_original_id[rig_group.ref_sensor_original_id]
        rig = pycolmap.Rig({"rig_id": rig_group.mapped_id})
        rig.add_ref_sensor(
            pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=ref_sensor.mapped_id)
        )
        for original_sensor_id in rig_group.sensor_original_ids:
            sensor = rig_model.sensors_by_original_id[original_sensor_id]
            if sensor.mapped_id == ref_sensor.mapped_id:
                continue
            rig.add_sensor(
                pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=sensor.mapped_id),
                colmap_pose_to_rigid3d(
                    sensor.sensor_from_rig_qvec,
                    sensor.sensor_from_rig_tvec,
                ),
            )
        reconstruction.add_rig(rig)

    bindings_by_name = {binding.name: binding for binding in bindings}

    for frame in rig_model.frames:
        rig_group = rig_model.rigs_by_key[frame.rig_key]
        reconstruction.add_frame(
            pycolmap.Frame({"frame_id": frame.mapped_id, "rig_id": rig_group.mapped_id})
        )
        for image_name in frame.image_names:
            binding = bindings_by_name[image_name]
            rig_image = rig_model.images_by_name[image_name]
            if binding.camera_id != rig_image.mapped_camera_id:
                raise RuntimeError(
                    f"Camera id mismatch for {image_name}: "
                    f"database={binding.camera_id}, model={rig_image.mapped_camera_id}"
                )
            if binding.sensor_id != rig_image.mapped_camera_id:
                raise RuntimeError(
                    f"Sensor id mismatch for {image_name}: "
                    f"database={binding.sensor_id}, model={rig_image.mapped_camera_id}"
                )

            reconstruction.frame(frame.mapped_id).add_data_id(
                pycolmap.data_t(
                    sensor_id=pycolmap.sensor_t(
                        type=pycolmap.SensorType.CAMERA,
                        id=binding.sensor_id,
                    ),
                    id=binding.data_id,
                )
            )

            reconstruction.add_image(
                pycolmap.Image(
                    {
                        "image_id": binding.image_id,
                        "camera_id": binding.camera_id,
                        "name": binding.name,
                        "frame_id": frame.mapped_id,
                        "data_id": {
                            "sensor_id": {
                                "type": pycolmap.SensorType.CAMERA,
                                "id": binding.sensor_id,
                            },
                            "id": binding.data_id,
                        },
                    }
                )
            )

        pose_image = rig_model.images_by_name[frame.pose_image_name]
        reconstruction.frame(frame.mapped_id).set_cam_from_world(
            pose_image.mapped_camera_id,
            colmap_pose_to_rigid3d(pose_image.qvec, pose_image.tvec),
        )
        reconstruction.register_frame(frame.mapped_id)

    return reconstruction


def write_reconstruction_outputs(
    reconstruction: pycolmap.Reconstruction,
    binary_dir: Path,
    text_dir: Path,
) -> None:
    ensure_clean_dir(binary_dir)
    ensure_clean_dir(text_dir)
    reconstruction.write(str(binary_dir))
    reconstruction.write_text(str(text_dir))


def ensure_text_model_dir(model_dir: Path, filenames: Iterable[str]) -> Path:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing text model dir: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    missing = [str(name) for name in filenames if not (model_dir / str(name)).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete text model in {model_dir}. Missing files: {', '.join(missing)}"
        )
    return model_dir


def _copy_directory_contents(source_dir: Path, target_dir: Path) -> None:
    source_dir = source_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    for source_path in source_dir.iterdir():
        target_path = target_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(str(source_path), str(target_path), symlinks=False)
        else:
            copy_file_compatible(source_path, target_path)


def resolve_published_points3d_path(rig_dir: Path, fallback_path: Path) -> Path:
    workspace_raw_points3d_path = rig_dir.parent / "raw" / "0" / "points3D.txt"
    if workspace_raw_points3d_path.is_file():
        return workspace_raw_points3d_path.resolve()

    fallback_path = fallback_path.expanduser().resolve()
    if fallback_path.is_file():
        return fallback_path

    raise FileNotFoundError(
        "Could not resolve points3D.txt for publishing. Checked "
        f"{workspace_raw_points3d_path} and {fallback_path}."
    )


def summarize_reconstruction(reconstruction: pycolmap.Reconstruction) -> Dict[str, float | int]:
    return {
        "num_rigs": int(reconstruction.num_rigs()),
        "num_cameras": int(reconstruction.num_cameras()),
        "num_frames": int(reconstruction.num_frames()),
        "num_reg_frames": int(reconstruction.num_reg_frames()),
        "num_images": int(reconstruction.num_images()),
        "num_reg_images": int(reconstruction.num_reg_images()),
        "num_points3D": int(reconstruction.num_points3D()),
        "num_observations": int(reconstruction.compute_num_observations()),
        "mean_track_length": float(reconstruction.compute_mean_track_length()),
        "mean_observations_per_image": float(
            reconstruction.compute_mean_observations_per_reg_image()
        ),
        "mean_reprojection_error": float(reconstruction.compute_mean_reprojection_error()),
    }


def run_bundle_adjustment(
    triangulated_model_dir: Path,
    output_binary_dir: Path,
    output_text_dir: Path,
    use_gpu: bool,
    refine_focal_length: bool,
    refine_principal_point: bool,
    refine_extra_params: bool,
    refine_rig_from_world: bool,
    refine_sensor_from_rig: bool,
) -> Dict[str, float | int]:
    reconstruction = pycolmap.Reconstruction(str(triangulated_model_dir))
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = bool(refine_focal_length)
    options.refine_principal_point = bool(refine_principal_point)
    options.refine_extra_params = bool(refine_extra_params)
    options.refine_rig_from_world = bool(refine_rig_from_world)
    options.refine_sensor_from_rig = bool(refine_sensor_from_rig)
    if hasattr(options, "use_gpu"):
        options.use_gpu = bool(use_gpu)
    elif hasattr(options, "ceres") and hasattr(options.ceres, "use_gpu"):
        options.ceres.use_gpu = bool(use_gpu)

    pycolmap.bundle_adjustment(reconstruction, options)
    reconstruction.update_point_3d_errors()
    write_reconstruction_outputs(reconstruction, output_binary_dir, output_text_dir)
    return summarize_reconstruction(reconstruction)


def publish_bundle_adjusted_sparse_model(
    rig_dir: Path,
    ba_text_dir: Path,
    raw_points3d_path: Path,
) -> tuple[Path, Path]:
    source_sparse_dir = rig_dir / "sparse" / "0"
    backup_sparse_dir = rig_dir / "sparse_bak" / "0"
    published_sparse_dir = rig_dir / "sparse" / "0"
    ba_text_dir = ensure_text_model_dir(ba_text_dir, RIG_SYNC_TEXT_FILES)
    raw_points3d_path = raw_points3d_path.expanduser().resolve()

    if not source_sparse_dir.exists():
        raise FileNotFoundError(f"Source sparse model not found: {source_sparse_dir}")
    if not raw_points3d_path.is_file():
        raise FileNotFoundError(f"Raw points3D.txt not found: {raw_points3d_path}")
    if backup_sparse_dir.exists():
        shutil.rmtree(backup_sparse_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir = published_sparse_dir.parent / f".{published_sparse_dir.name}_staging_{timestamp}"
    if staging_dir.exists():
        raise FileExistsError(f"Staging sparse dir already exists: {staging_dir}")

    backup_sparse_dir.parent.mkdir(parents=True, exist_ok=True)
    source_sparse_dir.rename(backup_sparse_dir)
    try:
        _copy_directory_contents(ba_text_dir, staging_dir)
        copy_file_compatible(raw_points3d_path, staging_dir / "points3D.txt")
        staging_dir.rename(published_sparse_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if backup_sparse_dir.exists() and not published_sparse_dir.exists():
            backup_sparse_dir.rename(published_sparse_dir)
        raise

    return backup_sparse_dir, published_sparse_dir


def resolve_vocab_tree_path(args: argparse.Namespace) -> Path | None:
    if args.vocab_tree_path:
        return Path(args.vocab_tree_path).expanduser().resolve()
    default_path = Path(__file__).resolve().parent / "vocab.bin"
    return default_path if default_path.exists() else None


def sensor_summary(rig_model: RigModel) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for sensor in rig_model.sensors:
        summary[str(sensor.original_id)] = {
            "rig_key": sensor.rig_key,
            "mapped_camera_id": int(sensor.mapped_id),
            "num_observations": int(sensor.num_observations),
            "sensor_from_rig_qvec": [float(value) for value in sensor.sensor_from_rig_qvec.tolist()],
            "sensor_from_rig_tvec": [float(value) for value in sensor.sensor_from_rig_tvec.tolist()],
        }
    return summary


def rig_summary(rig_model: RigModel) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for rig_group in rig_model.rigs:
        summary[rig_group.rig_key] = {
            "mapped_rig_id": int(rig_group.mapped_id),
            "ref_sensor_original_id": int(rig_group.ref_sensor_original_id),
            "sensor_original_ids": [int(value) for value in rig_group.sensor_original_ids],
        }
    return summary


def main() -> None:
    args = parse_args()
    rig_dir = Path(args.rig_dir).expanduser().resolve()
    workspace_dir = (
        Path(args.workspace_dir).expanduser().resolve()
        if args.workspace_dir
        else (rig_dir / "known_pose_ba")
    )
    feature_cache_db_path = (
        Path(args.feature_cache_db).expanduser().resolve()
        if args.feature_cache_db
        else default_feature_cache_db_path(rig_dir)
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

    rig_model = load_rig_model(
        rig_dir,
        share_intrinsics_across_rigs=args.share_intrinsics_across_rigs,
        frame_center_tolerance=float(args.frame_center_tolerance),
        frame_grouping=args.frame_grouping,
    )
    image_list_path = workspace_dir / "image_list.txt"
    write_image_list(rig_model, image_list_path)

    ltg_artifacts = None
    reuse_existing_ltg_matches = False
    if args.ltg:
        if feature_cache_db_path.exists():
            print(
                f"LTG mode ignores the COLMAP feature cache database: {feature_cache_db_path}",
                flush=True,
            )
        ltg_utils_dir = Path(__file__).resolve().parent / "metacam_utils"
        if str(ltg_utils_dir) not in sys.path:
            sys.path.insert(0, str(ltg_utils_dir))
        from ltg_utils import LtgPairingConfig, find_existing_ltg_artifacts, inspect_existing_ltg_database
        database_path = workspace_dir / "database.db"
        reuse_existing_ltg_matches, reuse_reason = inspect_existing_ltg_database(
            database_path=database_path,
            expected_image_names=[record.name for record in rig_model.images],
        )
        if reuse_existing_ltg_matches:
            print(f"Reusing existing LTG database from {database_path}: {reuse_reason}", flush=True)
            ltg_artifacts = find_existing_ltg_artifacts(
                workspace_dir / "ltg",
                expected_image_names=[record.name for record in rig_model.images],
                expected_pairing=LtgPairingConfig(
                    local_window=int(args.ltg_local_window),
                    global_every=int(args.ltg_global_every),
                    netvlad_num_matched=int(args.ltg_netvlad_num_matched),
                    spatial_num_neighbors=int(args.ltg_spatial_num_neighbors),
                    spatial_max_distance=float(args.ltg_spatial_max_distance),
                ),
            )
        else:
            if database_path.exists():
                print(f"Rebuilding LTG database at {database_path}: {reuse_reason}", flush=True)
                database_path.unlink()
            run_command(
                ["colmap", "database_creator", "--database_path", str(database_path)],
                env=env,
            )
            import_images_for_ltg_database(rig_model, database_path)
    else:
        database_path = create_database(
            rig_model=rig_model,
            workspace_dir=workspace_dir,
            use_gpu=args.use_gpu,
            image_list_path=image_list_path,
            env=env,
            feature_cache_db_path=feature_cache_db_path,
        )

    if args.ltg:
        print("Using LTG matching; --matcher/--vocab-tree-path/--spatial-* are ignored.", flush=True)
        if not reuse_existing_ltg_matches:
            ltg_artifacts = run_ltg_matching(
                args=args,
                rig_model=rig_model,
                database_path=database_path,
                workspace_dir=workspace_dir,
                env=env,
            )
        matcher_used = "lightglue"
        patch_database_with_rig_model(database_path, rig_model)
        populate_database_sensor_from_rig(database_path, rig_model)
        num_pose_priors = write_pose_priors(database_path, rig_model)
    else:
        patch_database_with_rig_model(database_path, rig_model)
        populate_database_sensor_from_rig(database_path, rig_model)
        num_pose_priors = write_pose_priors(database_path, rig_model)
        matcher_used = run_matcher(
            database_path=database_path,
            matcher=args.matcher,
            vocab_tree_path=resolve_vocab_tree_path(args),
            use_gpu=args.use_gpu,
            spatial_ignore_z=args.spatial_ignore_z,
            spatial_max_num_neighbors=args.spatial_max_num_neighbors,
            spatial_min_num_neighbors=args.spatial_min_num_neighbors,
            spatial_max_distance=args.spatial_max_distance,
            env=env,
        )
    matching_summary = summarize_database_matching(database_path)
    print(
        "Database matching summary: "
        f"{matching_summary['num_verified_image_pairs']} verified pairs, "
        f"{matching_summary['num_inlier_matches']} inlier matches",
        flush=True,
    )

    bindings = load_database_bindings(database_path)
    init_reconstruction = build_known_pose_reconstruction(rig_model, bindings)
    init_binary_dir = workspace_dir / "init_model"
    init_text_dir = workspace_dir / "init_model_txt"
    write_reconstruction_outputs(init_reconstruction, init_binary_dir, init_text_dir)

    triangulated_binary_dir = workspace_dir / "triangulated"
    triangulated_text_dir = workspace_dir / "triangulated_txt"
    ensure_clean_dir(triangulated_binary_dir)
    triangulated = pycolmap.triangulate_points(
        init_reconstruction,
        str(database_path),
        str(rig_model.images_dir),
        str(triangulated_binary_dir),
        clear_points=True,
        refine_intrinsics=False,
    )
    ensure_clean_dir(triangulated_text_dir)
    triangulated.write_text(str(triangulated_text_dir))
    triangulation_summary = summarize_reconstruction(triangulated)
    print(f"Triangulated points: {triangulation_summary['num_points3D']}", flush=True)

    ba_binary_dir = workspace_dir / "bundle_adjusted"
    ba_text_dir = workspace_dir / "bundle_adjusted_txt"
    ba_summary = run_bundle_adjustment(
        triangulated_model_dir=triangulated_binary_dir,
        output_binary_dir=ba_binary_dir,
        output_text_dir=ba_text_dir,
        use_gpu=args.use_gpu,
        refine_focal_length=args.refine_focal_length,
        refine_principal_point=args.refine_principal_point,
        refine_extra_params=args.refine_extra_params,
        refine_rig_from_world=args.refine_rig_from_world,
        refine_sensor_from_rig=args.refine_sensor_from_rig,
    )
    raw_points3d_path = resolve_published_points3d_path(
        rig_dir=rig_dir,
        fallback_path=triangulated_text_dir / "points3D.txt",
    )
    backup_sparse_dir, published_sparse_dir = publish_bundle_adjusted_sparse_model(
        rig_dir=rig_dir,
        ba_text_dir=ba_text_dir,
        raw_points3d_path=raw_points3d_path,
    )

    summary = {
        "rig_dir": str(rig_dir),
        "workspace_dir": str(workspace_dir),
        "feature_cache_db_path": str(feature_cache_db_path),
        "database_path": str(database_path),
        "matcher": matcher_used,
        "ltg": bool(args.ltg),
        "ltg_work_dir": str(ltg_artifacts.work_dir) if ltg_artifacts is not None else None,
        "ltg_pairs_path": str(ltg_artifacts.pairs_path) if ltg_artifacts is not None else None,
        "ltg_features_path": str(ltg_artifacts.features_path) if ltg_artifacts is not None else None,
        "ltg_matches_path": str(ltg_artifacts.matches_path) if ltg_artifacts is not None else None,
        "ltg_pair_count": int(ltg_artifacts.num_pairs) if ltg_artifacts is not None else None,
        "num_pose_priors": int(num_pose_priors),
        "grouping_mode": rig_model.grouping_mode,
        "num_rigs": int(len(rig_model.rigs)),
        "share_intrinsics_across_rigs": bool(args.share_intrinsics_across_rigs),
        "frame_center_tolerance": float(args.frame_center_tolerance),
        "frame_grouping": str(args.frame_grouping),
        "refine_rig_from_world": bool(args.refine_rig_from_world),
        "refine_sensor_from_rig": bool(args.refine_sensor_from_rig),
        "spatial_matching_options": {
            "ignore_z": bool(args.spatial_ignore_z),
            "max_num_neighbors": int(args.spatial_max_num_neighbors),
            "min_num_neighbors": int(args.spatial_min_num_neighbors),
            "max_distance": float(args.spatial_max_distance),
        },
        "num_inferred_frames": int(len(rig_model.frames)),
        "backup_sparse_dir": str(backup_sparse_dir),
        "published_sparse_dir": str(published_sparse_dir),
        "bundle_adjusted_text_dir": str(ba_text_dir),
        "published_points3d_source": str(raw_points3d_path),
        "camera_id_map": {
            str(original_id): mapped_ids
            for original_id, mapped_ids in rig_model.camera_id_map.items()
        },
        "rigs": rig_summary(rig_model),
        "sensors": sensor_summary(rig_model),
        "matching_database": matching_summary,
        "triangulation": triangulation_summary,
        "bundle_adjustment": ba_summary,
    }
    summary_path = workspace_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Bundle-adjusted points: {ba_summary['num_points3D']}", flush=True)
    print(f"Inferred rigs: {len(rig_model.rigs)}", flush=True)
    print(f"Inferred frames: {len(rig_model.frames)}", flush=True)
    print(f"Frame grouping mode: {rig_model.grouping_mode}", flush=True)
    print(f"Triangulated model: {triangulated_binary_dir}", flush=True)
    print(f"Backed up original sparse model to: {backup_sparse_dir}", flush=True)
    print(f"Published bundle-adjusted model to: {published_sparse_dir}", flush=True)
    print(f"Bundle-adjusted text model: {ba_text_dir}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
