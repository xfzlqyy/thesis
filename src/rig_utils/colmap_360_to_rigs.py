"""Convert COLMAP fisheye images and sparse poses to rig views and COLMAP model."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from _project_root import ensure_project_root_on_path
from fs_compat import copy_file_compatible
from rig_view_utils import build_perspective_k, default_rig_rotations

PROJECT_ROOT = ensure_project_root_on_path(__file__)

from lib.utils.colmap_utils import (  # noqa: E402
    Camera,
    Image,
    qvec2rotmat,
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    rotmat2qvec,
    write_cameras_text,
    write_images_text,
)
from lib.utils.progress_utils import tqdm  # noqa: E402

OPENCV_FISHEYE_MODEL_NAME = "OPENCV_FISHEYE"
RAD_TAN_THIN_PRISM_FISHEYE_MODEL_NAME = "RAD_TAN_THIN_PRISM_FISHEYE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        default=None,
        help="Workspace root containing images/, sparse/0/, and rigs/.",
    )
    parser.add_argument("--images-dir", default=None, help="Filtered fisheye images dir.")
    parser.add_argument("--colmap-dir", default=None, help="COLMAP sparse/0 dir.")
    parser.add_argument("--output-dir", default=None, help="Output rig root dir.")
    parser.add_argument(
        "--max-images",
        "--num-images",
        dest="max_images",
        type=int,
        default=-1,
        help="Maximum number of COLMAP images to convert. Use -1 to convert all images.",
    )
    parser.add_argument("--fov-degrees", type=float, default=60.0, help="Perspective FOV.")
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        default=(800, 800),
        metavar=("W", "H"),
        help="Perspective output size (W H).",
    )
    return parser.parse_args()


def resolve_io_dirs(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    data_root = None
    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()

    images_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else None
    colmap_dir = Path(args.colmap_dir).expanduser().resolve() if args.colmap_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    if data_root is not None:
        images_dir = images_dir or (data_root / "images")
        colmap_dir = colmap_dir or (data_root / "sparse" / "0")
        output_dir = output_dir or (data_root / "rigs")

    missing = []
    if images_dir is None:
        missing.append("--images-dir")
    if colmap_dir is None:
        missing.append("--colmap-dir")
    if output_dir is None:
        missing.append("--output-dir")

    if missing:
        raise ValueError(
            "Missing required paths: "
            + ", ".join(missing)
            + ". Provide them explicitly, or set --data-root to infer them."
        )

    return images_dir, colmap_dir, output_dir


def require_pycolmap():
    try:
        import pycolmap
    except ImportError as exc:
        raise ImportError(
            "RAD_TAN_THIN_PRISM_FISHEYE rig export requires pycolmap."
        ) from exc
    return pycolmap


def scaled_camera_params(camera: Camera, image_shape: Tuple[int, int]) -> np.ndarray:
    if camera.width <= 0 or camera.height <= 0:
        raise ValueError(f"COLMAP camera {camera.id} has invalid image size: {camera.width}x{camera.height}")

    params = np.asarray(camera.params, dtype=np.float64).reshape(-1).copy()
    if params.shape[0] < 4:
        raise ValueError(
            f"COLMAP camera {camera.id} has invalid intrinsics params: {params}"
        )

    img_h, img_w = image_shape
    sx = img_w / float(camera.width)
    sy = img_h / float(camera.height)
    params[0] *= sx
    params[1] *= sy
    params[2] *= sx
    params[3] *= sy
    return params


def scaled_fisheye_intrinsics(camera: Camera, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    model_name = str(camera.model)
    if model_name != OPENCV_FISHEYE_MODEL_NAME:
        raise ValueError(
            f"Unsupported OpenCV fisheye rig export camera model: {model_name}. "
            f"Expected {OPENCV_FISHEYE_MODEL_NAME}."
        )

    params = scaled_camera_params(camera, image_shape)
    if params.shape[0] < 8:
        raise ValueError(
            f"COLMAP camera {camera.id} has invalid {OPENCV_FISHEYE_MODEL_NAME} params: {params}"
        )

    fx, fy, cx, cy = params[:4]
    dist = params[4:8]

    k = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return k, dist.copy()


def make_pycolmap_camera(camera: Camera, image_shape: Tuple[int, int]):
    pycolmap = require_pycolmap()
    model_name = str(camera.model)
    try:
        model_id = getattr(pycolmap.CameraModelId, model_name)
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported COLMAP camera model for generic rig export: {model_name}"
        ) from exc

    params = scaled_camera_params(camera, image_shape)
    img_h, img_w = image_shape
    return pycolmap.Camera(
        {
            "camera_id": int(camera.id),
            "model": model_id,
            "width": int(img_w),
            "height": int(img_h),
            "params": params.tolist(),
        }
    )


def build_generic_maps(
    source_camera,
    k_perspective: np.ndarray,
    output_size: Tuple[int, int],
    rotations: Dict[str, np.ndarray],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    width, height = output_size
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    rectified_cam_points = np.stack(
        [
            (grid_x - float(k_perspective[0, 2])) / float(k_perspective[0, 0]),
            (grid_y - float(k_perspective[1, 2])) / float(k_perspective[1, 1]),
            np.ones_like(grid_x),
        ],
        axis=-1,
    ).reshape(-1, 3)

    maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, rmat in rotations.items():
        source_cam_points = rectified_cam_points @ np.asarray(rmat, dtype=np.float64)
        source_pixels = np.asarray(source_camera.img_from_cam(source_cam_points), dtype=np.float64)
        map_x = np.full((height * width,), -1.0, dtype=np.float32)
        map_y = np.full((height * width,), -1.0, dtype=np.float32)
        valid_mask = np.isfinite(source_pixels).all(axis=1)
        if np.any(valid_mask):
            map_x[valid_mask] = source_pixels[valid_mask, 0].astype(np.float32, copy=False)
            map_y[valid_mask] = source_pixels[valid_mask, 1].astype(np.float32, copy=False)
        maps[name] = (map_x.reshape(height, width), map_y.reshape(height, width))
    return maps


def build_maps(
    camera: Camera,
    image_shape: Tuple[int, int],
    output_size: Tuple[int, int],
    fov_degrees: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    k_perspective = build_perspective_k(output_size, fov_degrees)
    rotations = default_rig_rotations()

    model_name = str(camera.model)
    if model_name == OPENCV_FISHEYE_MODEL_NAME:
        k, d = scaled_fisheye_intrinsics(camera, image_shape)
        maps = {}
        for name, rmat in rotations.items():
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                k, d, rmat, k_perspective, output_size, cv2.CV_16SC2
            )
            maps[name] = (map1, map2)
    elif model_name == RAD_TAN_THIN_PRISM_FISHEYE_MODEL_NAME:
        source_camera = make_pycolmap_camera(camera, image_shape)
        maps = build_generic_maps(source_camera, k_perspective, output_size, rotations)
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model for fisheye rig export: {model_name}. "
            f"Expected {OPENCV_FISHEYE_MODEL_NAME} or {RAD_TAN_THIN_PRISM_FISHEYE_MODEL_NAME}."
        )
    return k_perspective, rotations, maps


def normalize_colmap_image_name(image_name: str) -> str:
    for prefix in ("left__", "right__"):
        if image_name.startswith(prefix):
            return image_name.replace("__", "/", 1)
    return image_name


def normalize_colmap_images(colmap_images: Dict[int, Image]) -> Dict[int, Image]:
    normalized_images: Dict[int, Image] = {}
    for image_id, image in colmap_images.items():
        normalized_name = normalize_colmap_image_name(image.name)
        if normalized_name == image.name:
            normalized_images[image_id] = image
            continue

        normalized_images[image_id] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=normalized_name,
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        )
    return normalized_images


def read_colmap_images(colmap_dir: Path):
    bin_path = colmap_dir / "images.bin"
    txt_path = colmap_dir / "images.txt"
    if bin_path.exists():
        return normalize_colmap_images(read_images_binary(str(bin_path)))
    if txt_path.exists():
        return normalize_colmap_images(read_images_text(str(txt_path)))
    raise FileNotFoundError(f"No COLMAP images file found in {colmap_dir}")


def read_colmap_cameras(colmap_dir: Path):
    bin_path = colmap_dir / "cameras.bin"
    txt_path = colmap_dir / "cameras.txt"
    if bin_path.exists():
        return read_cameras_binary(str(bin_path))
    if txt_path.exists():
        return read_cameras_text(str(txt_path))
    raise FileNotFoundError(f"No COLMAP cameras file found in {colmap_dir}")


def infer_cam_label(image_name: str, camera_id: int) -> str:
    parts = Path(image_name).parts
    for part in reversed(parts[:-1]):
        lower_part = part.lower()
        if lower_part in {"left", "right"} or lower_part.startswith("cam"):
            return part

    stem = Path(image_name).stem
    stem_lower = stem.lower()
    for prefix in ("left_", "right_", "cam0_", "cam1_", "cam2_", "cam3_"):
        if stem_lower.startswith(prefix):
            return prefix[:-1]
    return f"cam{camera_id}"


def main() -> None:
    args = parse_args()
    if args.max_images < -1:
        raise ValueError(f"--max-images must be -1 or >= 0, got {args.max_images}")

    images_dir, colmap_dir, output_dir = resolve_io_dirs(args)

    colmap_cameras = read_colmap_cameras(colmap_dir)
    if not colmap_cameras:
        raise RuntimeError(f"No cameras found in COLMAP model: {colmap_dir}")
    colmap_images = read_colmap_images(colmap_dir)
    if not colmap_images:
        raise RuntimeError(f"No images found in COLMAP model: {colmap_dir}")

    output_images_dir = output_dir / "images"
    output_sparse_dir = output_dir / "sparse" / "0"
    output_sparse_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    map_cache: Dict[
        Tuple[int, int, int],
        Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]],
    ] = {}
    camera_ids: Dict[Tuple[int, str], int] = {}
    cameras: Dict[int, Camera] = {}
    images: Dict[int, Image] = {}
    next_camera_id = 1
    next_image_id = 1

    sorted_images = sorted(colmap_images.items(), key=lambda item: item[0])
    total_images = len(sorted_images)
    if args.max_images >= 0:
        sorted_images = sorted_images[: args.max_images]
        print(
            f"Limiting rig conversion to first {len(sorted_images)} "
            f"of {total_images} COLMAP images."
        )

    for _, img in tqdm(
        sorted_images,
        total=len(sorted_images),
        desc="Converting rig views",
        disable=not sys.stdout.isatty(),
    ):
        image_path = images_dir / img.name
        if not image_path.exists():
            print(f"Warning: missing image {image_path}")
            continue

        base_image = cv2.imread(str(image_path))
        if base_image is None:
            print(f"Warning: failed to read {image_path}")
            continue

        source_camera = colmap_cameras.get(img.camera_id)
        if source_camera is None:
            print(f"Warning: missing camera {img.camera_id} for image {img.name}")
            continue

        cam_label = infer_cam_label(img.name, img.camera_id)
        cache_key = (img.camera_id, base_image.shape[1], base_image.shape[0])
        if cache_key not in map_cache:
            k_perspective, rotations, maps = build_maps(
                source_camera,
                base_image.shape[:2],
                tuple(args.output_size),
                args.fov_degrees,
            )
            map_cache[cache_key] = (k_perspective, rotations, maps)

        k_perspective, rotations, maps = map_cache[cache_key]

        R_base = qvec2rotmat(img.qvec)
        t_base = np.array(img.tvec)

        frame_stem = Path(img.name).stem
        frame_ext = Path(img.name).suffix or ".jpg"

        for view_name, (map1, map2) in maps.items():
            view_key = (img.camera_id, view_name)
            view_image = cv2.remap(
                base_image,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            out_name = f"{cam_label}_{view_name}_{frame_stem}{frame_ext}"
            out_path = output_images_dir / out_name
            cv2.imwrite(str(out_path), view_image)

            if view_key not in camera_ids:
                camera_id = next_camera_id
                camera_ids[view_key] = camera_id
                params = [
                    float(k_perspective[0, 0]),
                    float(k_perspective[1, 1]),
                    float(k_perspective[0, 2]),
                    float(k_perspective[1, 2]),
                ]
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model="PINHOLE",
                    width=int(args.output_size[0]),
                    height=int(args.output_size[1]),
                    params=params,
                )
                next_camera_id += 1
            else:
                camera_id = camera_ids[view_key]

            R_view = rotations[view_name] @ R_base
            t_view = rotations[view_name] @ t_base
            qvec = rotmat2qvec(R_view)

            rel_name = out_name
            images[next_image_id] = Image(
                id=next_image_id,
                qvec=qvec,
                tvec=t_view,
                camera_id=camera_id,
                name=rel_name,
                xys=[],
                point3D_ids=[],
            )
            next_image_id += 1

    write_cameras_text(cameras, str(output_sparse_dir / "cameras.txt"))
    write_images_text(images, str(output_sparse_dir / "images.txt"))

    points3d_txt = colmap_dir / "points3D.txt"
    if points3d_txt.exists():
        copy_file_compatible(points3d_txt, output_sparse_dir / "points3D.txt")
    else:
        print(f"Warning: missing points3D.txt in {colmap_dir}, skipping copy.")

    print(
        f"Generated rigs in {output_dir} "
        f"({len(cameras)} cameras, {len(images)} images). "
    )


if __name__ == "__main__":
    main()
