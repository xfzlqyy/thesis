"""Fast validation for COLMAP rig grouping and sparse pose constraints."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "lib").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError(f"Could not locate project root for {current}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.utils.colmap_utils import qvec2rotmat, read_images_binary, read_images_text  # noqa: E402


@dataclass(frozen=True)
class ExpectedSensor:
    name: str
    image_prefix: str
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class ExpectedRig:
    index: int
    sensors: Dict[str, ExpectedSensor]
    ref_sensor_name: str

    @property
    def sensor_names(self) -> set[str]:
        return set(self.sensors.keys())


@dataclass(frozen=True)
class ImageMatch:
    image_id: int
    name: str
    rig_index: int
    sensor_name: str
    leaf_name: str


@dataclass(frozen=True)
class SparseFrame:
    frame_id: int
    rig_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    data: List[Tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="COLMAP database.db path.")
    parser.add_argument("--rig-config", required=True, help="COLMAP rig_config.json path.")
    parser.add_argument(
        "--sparse-dir",
        default=None,
        help="Optional COLMAP sparse/0 dir. If set, also verify sparse poses.",
    )
    parser.add_argument(
        "--center-tol",
        type=float,
        default=1e-4,
        help="Maximum allowed camera-center error in sparse pose validation.",
    )
    parser.add_argument(
        "--rotation-tol-deg",
        type=float,
        default=1e-3,
        help="Maximum allowed rotation error in degrees for sparse pose validation.",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=10,
        help="Maximum number of failures to print per validation stage.",
    )
    return parser.parse_args()


def load_rig_config(path: Path) -> List[ExpectedRig]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    expected_rigs: List[ExpectedRig] = []
    for rig_index, rig in enumerate(payload):
        sensors: Dict[str, ExpectedSensor] = {}
        ref_sensor_name: str | None = None
        for camera in rig.get("cameras", []):
            image_prefix = str(camera["image_prefix"])
            sensor_name = image_prefix[:-1] if image_prefix.endswith(("/", "_")) else image_prefix
            if "cam_from_rig_rotation" in camera:
                rotation = qvec2rotmat(np.asarray(camera["cam_from_rig_rotation"], dtype=np.float64))
            else:
                rotation = np.eye(3, dtype=np.float64)
            translation = np.asarray(
                camera.get("cam_from_rig_translation", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            sensors[sensor_name] = ExpectedSensor(
                name=sensor_name,
                image_prefix=image_prefix,
                rotation=rotation,
                translation=translation,
            )
            if camera.get("ref_sensor"):
                ref_sensor_name = sensor_name
        if not sensors:
            raise ValueError(f"Rig {rig_index} has no sensors in {path}")
        if ref_sensor_name is None:
            raise ValueError(f"Rig {rig_index} has no ref_sensor in {path}")
        expected_rigs.append(ExpectedRig(index=rig_index, sensors=sensors, ref_sensor_name=ref_sensor_name))
    if not expected_rigs:
        raise ValueError(f"No rigs found in {path}")
    return expected_rigs


def match_image_name(name: str, expected_rigs: Sequence[ExpectedRig]) -> ImageMatch:
    matches: List[Tuple[int, ExpectedSensor]] = []
    for rig in expected_rigs:
        for sensor in rig.sensors.values():
            if name.startswith(sensor.image_prefix):
                matches.append((len(sensor.image_prefix), sensor))

    if not matches:
        raise ValueError(f"Image '{name}' does not match any image_prefix in rig_config.")

    matches.sort(key=lambda item: item[0], reverse=True)
    longest = matches[0][0]
    winners = [sensor for prefix_len, sensor in matches if prefix_len == longest]
    if len(winners) != 1:
        raise ValueError(f"Image '{name}' matches multiple rig prefixes with same length.")

    sensor = winners[0]
    rig_index = next(rig.index for rig in expected_rigs if sensor.name in rig.sensors)
    leaf_name = name[len(sensor.image_prefix) :]
    return ImageMatch(
        image_id=-1,
        name=name,
        rig_index=rig_index,
        sensor_name=sensor.name,
        leaf_name=leaf_name,
    )


def load_images_from_db(conn: sqlite3.Connection, expected_rigs: Sequence[ExpectedRig]) -> Dict[int, ImageMatch]:
    rows = conn.execute("SELECT image_id, name FROM images").fetchall()
    image_matches: Dict[int, ImageMatch] = {}
    for image_id, name in rows:
        matched = match_image_name(str(name), expected_rigs)
        image_matches[int(image_id)] = ImageMatch(
            image_id=int(image_id),
            name=matched.name,
            rig_index=matched.rig_index,
            sensor_name=matched.sensor_name,
            leaf_name=matched.leaf_name,
        )
    return image_matches


def validate_database(
    database_path: Path,
    expected_rigs: Sequence[ExpectedRig],
    max_findings: int,
) -> Tuple[Dict[int, Dict[int, str]], Dict[int, int]]:
    conn = sqlite3.connect(str(database_path))
    try:
        rig_ids = [int(row[0]) for row in conn.execute("SELECT rig_id FROM rigs ORDER BY rig_id").fetchall()]
        if len(rig_ids) != len(expected_rigs):
            raise ValueError(
                f"Database rig count mismatch: expected {len(expected_rigs)}, found {len(rig_ids)} ({rig_ids})"
            )

        image_matches = load_images_from_db(conn, expected_rigs)
        rows = conn.execute(
            """
            SELECT frames.frame_id, frames.rig_id, frame_data.sensor_id, frame_data.data_id
            FROM frames
            JOIN frame_data ON frames.frame_id = frame_data.frame_id
            ORDER BY frames.frame_id, frame_data.sensor_id
            """
        ).fetchall()
        if not rows:
            raise ValueError("Database contains no frames/frame_data rows after rig_configurator.")

        frames: Dict[int, List[Tuple[int, int, int]]] = {}
        for frame_id, rig_id, sensor_id, data_id in rows:
            frames.setdefault(int(frame_id), []).append((int(rig_id), int(sensor_id), int(data_id)))

        findings: List[str] = []
        sensor_name_map: Dict[int, Dict[int, str]] = {}
        rig_expected_map: Dict[int, int] = {}
        for frame_id, entries in frames.items():
            rig_ids_in_frame = {rig_id for rig_id, _, _ in entries}
            if len(rig_ids_in_frame) != 1:
                findings.append(f"frame {frame_id}: multiple database rig_ids {sorted(rig_ids_in_frame)}")
                if len(findings) >= max_findings:
                    break
                continue

            db_rig_id = next(iter(rig_ids_in_frame))
            matches = [image_matches[data_id] for _, _, data_id in entries]
            rig_indices = {match.rig_index for match in matches}
            if len(rig_indices) != 1:
                findings.append(
                    f"frame {frame_id}: images come from multiple expected rigs "
                    f"{sorted(rig_indices)} ({[match.name for match in matches]})"
                )
                if len(findings) >= max_findings:
                    break
                continue

            expected_rig_index = next(iter(rig_indices))
            expected_rig = expected_rigs[expected_rig_index]
            sensor_names = {match.sensor_name for match in matches}
            leaf_names = {match.leaf_name for match in matches}

            if db_rig_id in rig_expected_map and rig_expected_map[db_rig_id] != expected_rig_index:
                findings.append(
                    f"db rig {db_rig_id}: mapped to multiple expected rigs "
                    f"{rig_expected_map[db_rig_id]} and {expected_rig_index}"
                )
            else:
                rig_expected_map[db_rig_id] = expected_rig_index

            if len(entries) != len(expected_rig.sensors):
                findings.append(
                    f"frame {frame_id}: expected {len(expected_rig.sensors)} images, found {len(entries)}"
                )
            if sensor_names != expected_rig.sensor_names:
                findings.append(
                    f"frame {frame_id}: sensor set mismatch, expected {sorted(expected_rig.sensor_names)}, "
                    f"found {sorted(sensor_names)}"
                )
            if len(leaf_names) != 1:
                findings.append(
                    f"frame {frame_id}: images are not grouped by same source name ({sorted(leaf_names)})"
                )

            current_map = sensor_name_map.setdefault(db_rig_id, {})
            for _, sensor_id, data_id in entries:
                sensor_name = image_matches[data_id].sensor_name
                if sensor_id in current_map and current_map[sensor_id] != sensor_name:
                    findings.append(
                        f"db rig {db_rig_id}: sensor_id {sensor_id} maps to both "
                        f"{current_map[sensor_id]} and {sensor_name}"
                    )
                else:
                    current_map[sensor_id] = sensor_name

            if len(findings) >= max_findings:
                break

        if findings:
            joined = "\n".join(f"- {item}" for item in findings[:max_findings])
            raise ValueError(f"Database rig validation failed:\n{joined}")

        return sensor_name_map, rig_expected_map
    finally:
        conn.close()


def read_sparse_images(sparse_dir: Path):
    txt_path = sparse_dir / "images.txt"
    bin_path = sparse_dir / "images.bin"
    if txt_path.exists():
        return read_images_text(str(txt_path))
    if bin_path.exists():
        return read_images_binary(str(bin_path))
    raise FileNotFoundError(f"No COLMAP images file found in {sparse_dir}")


def read_frames_text(path: Path) -> List[SparseFrame]:
    frames: List[SparseFrame] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            frame_id = int(elems[0])
            rig_id = int(elems[1])
            qvec = np.asarray([float(value) for value in elems[2:6]], dtype=np.float64)
            tvec = np.asarray([float(value) for value in elems[6:9]], dtype=np.float64)
            num_data = int(elems[9])
            data_elems = elems[10:]
            if len(data_elems) != num_data * 3:
                raise ValueError(f"Invalid frames.txt line for frame {frame_id}: wrong DATA_IDS length.")
            data: List[Tuple[int, int]] = []
            for idx in range(num_data):
                sensor_type = data_elems[idx * 3]
                if sensor_type != "CAMERA":
                    raise ValueError(f"Unsupported sensor type '{sensor_type}' in frame {frame_id}")
                sensor_id = int(data_elems[idx * 3 + 1])
                data_id = int(data_elems[idx * 3 + 2])
                data.append((sensor_id, data_id))
            frames.append(SparseFrame(frame_id=frame_id, rig_id=rig_id, qvec=qvec, tvec=tvec, data=data))
    return frames


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = a @ b.T
    cos_theta = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def camera_center(rmat: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return -rmat.T @ tvec


def validate_sparse(
    sparse_dir: Path,
    expected_rigs: Sequence[ExpectedRig],
    sensor_name_map: Dict[int, Dict[int, str]],
    rig_expected_map: Dict[int, int],
    center_tol: float,
    rotation_tol_deg: float,
    max_findings: int,
) -> None:
    frames_path = sparse_dir / "frames.txt"
    if not frames_path.exists():
        raise FileNotFoundError(f"No frames.txt found in {sparse_dir}")

    images = read_sparse_images(sparse_dir)
    frames = read_frames_text(frames_path)
    if not frames:
        raise ValueError(f"No sparse frames found in {frames_path}")

    findings: List[str] = []
    max_center_error = 0.0
    max_rotation_error = 0.0
    worst_center: str | None = None
    worst_rotation: str | None = None

    for frame in frames:
        if frame.rig_id not in sensor_name_map:
            findings.append(f"sparse frame {frame.frame_id}: rig_id {frame.rig_id} missing in database sensor map")
            if len(findings) >= max_findings:
                break
            continue

        expected_rig_index = rig_expected_map[frame.rig_id]
        expected_rig = expected_rigs[expected_rig_index]
        r_rig = qvec2rotmat(frame.qvec)
        t_rig = frame.tvec

        for sensor_id, data_id in frame.data:
            image = images.get(data_id)
            if image is None:
                findings.append(f"sparse frame {frame.frame_id}: missing image pose for data_id {data_id}")
                continue

            sensor_name = sensor_name_map[frame.rig_id].get(sensor_id)
            if sensor_name is None:
                findings.append(
                    f"sparse frame {frame.frame_id}: sensor_id {sensor_id} has no database sensor-name mapping"
                )
                continue

            expected_sensor = expected_rig.sensors[sensor_name]
            r_expected = expected_sensor.rotation @ r_rig
            t_expected = expected_sensor.rotation @ t_rig + expected_sensor.translation

            r_actual = qvec2rotmat(image.qvec)
            t_actual = np.asarray(image.tvec, dtype=np.float64)

            center_error = float(
                np.linalg.norm(camera_center(r_actual, t_actual) - camera_center(r_expected, t_expected))
            )
            rotation_error = rotation_error_deg(r_actual, r_expected)

            if center_error > max_center_error:
                max_center_error = center_error
                worst_center = f"frame {frame.frame_id}, image {image.name}"
            if rotation_error > max_rotation_error:
                max_rotation_error = rotation_error
                worst_rotation = f"frame {frame.frame_id}, image {image.name}"

            if center_error > center_tol:
                findings.append(
                    f"sparse frame {frame.frame_id}, image {image.name}: center error {center_error:.6e} "
                    f"> tol {center_tol:.6e}"
                )
            if rotation_error > rotation_tol_deg:
                findings.append(
                    f"sparse frame {frame.frame_id}, image {image.name}: rotation error {rotation_error:.6e} deg "
                    f"> tol {rotation_tol_deg:.6e} deg"
                )
            if len(findings) >= max_findings:
                break
        if len(findings) >= max_findings:
            break

    if findings:
        joined = "\n".join(f"- {item}" for item in findings[:max_findings])
        raise ValueError(
            "Sparse rig validation failed:\n"
            f"{joined}\n"
            f"max_center_error={max_center_error:.6e} ({worst_center})\n"
            f"max_rotation_error_deg={max_rotation_error:.6e} ({worst_rotation})"
        )

    print(
        "Sparse rig validation passed: "
        f"frames={len(frames)}, max_center_error={max_center_error:.6e} ({worst_center}), "
        f"max_rotation_error_deg={max_rotation_error:.6e} ({worst_rotation})"
    )


def main() -> None:
    args = parse_args()
    database_path = Path(args.database).expanduser().resolve()
    rig_config_path = Path(args.rig_config).expanduser().resolve()
    sparse_dir = Path(args.sparse_dir).expanduser().resolve() if args.sparse_dir else None

    expected_rigs = load_rig_config(rig_config_path)
    print(
        "Loaded rig config:",
        f"rigs={len(expected_rigs)}, sensors_per_rig={[len(rig.sensors) for rig in expected_rigs]}",
    )

    sensor_name_map, rig_expected_map = validate_database(
        database_path=database_path,
        expected_rigs=expected_rigs,
        max_findings=int(args.max_findings),
    )
    print(
        "Database rig validation passed:",
        f"rigs={len(sensor_name_map)}, sensor_ids_per_rig={[len(mapping) for mapping in sensor_name_map.values()]}",
    )

    if sparse_dir is not None:
        validate_sparse(
            sparse_dir=sparse_dir,
            expected_rigs=expected_rigs,
            sensor_name_map=sensor_name_map,
            rig_expected_map=rig_expected_map,
            center_tol=float(args.center_tol),
            rotation_tol_deg=float(args.rotation_tol_deg),
            max_findings=int(args.max_findings),
        )


if __name__ == "__main__":
    main()
