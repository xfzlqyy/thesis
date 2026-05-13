"""Iteratively re-triangulate and bundle-adjust rig known-pose reconstructions."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pycolmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.fs_compat import copy_file_compatible, resolve_image_root_for_names  # noqa: E402


STANDARD_MODEL_STEMS = ("cameras", "images")
RIG_MODEL_STEMS = STANDARD_MODEL_STEMS + ("rigs", "frames")
STANDARD_SYNC_TEXT_FILES = ("cameras.txt", "images.txt")
RIG_SYNC_TEXT_FILES = STANDARD_SYNC_TEXT_FILES + ("rigs.txt", "frames.txt")
CLI_DESCRIPTOR_COLUMNS = ("image_id", "rows", "cols", "data")
CLI_POSE_PRIOR_COLUMNS = ("image_id", "position", "coordinate_system", "position_covariance")


@dataclass(frozen=True)
class IterativeRigImage:
    name: str
    mapped_camera_id: int
    center: np.ndarray


@dataclass
class IterativeRigBaSummary:
    rig_dir: str
    run_root: str
    iteration_dir: str
    database_path: str
    input_model_dir: str
    raw_model_dir: str
    raw_text_dir: str
    output_model_dir: str
    output_text_dir: str
    no_rig_constraint: bool
    rerun_geometric_verifier: bool
    ltg: bool
    guided_matching: bool
    rig_verification: bool
    triangulator_fix_existing_frames: bool
    bundle_adjust_refine_rig_from_world: bool
    bundle_adjust_refine_sensor_from_rig: bool
    bundle_adjust_refine_intrinsics: bool
    ltg_work_dir: str | None
    ltg_pairs_path: str | None
    ltg_features_path: str | None
    ltg_matches_path: str | None
    ltg_pair_count: int | None
    point_triangulator_log_path: str
    bundle_adjuster_log_path: str
    geometric_verifier_log_path: str | None
    raw_model_converter_log_path: str
    output_model_converter_log_path: str
    model_analyzer_log_path: str
    num_images: int
    num_cameras: int
    raw_num_points3d: int
    raw_mean_track_length: float
    output_num_points3d: int
    output_mean_track_length: float
    num_observations: int | None
    mean_reprojection_error: float | None
    sync_latest_requested: bool
    sync_dry_run: bool
    synced_latest: bool
    sync_target_model_dir: str | None
    sync_points3d_path: str | None
    sync_backup_dir: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rig-dir",
        type=Path,
        required=True,
        help="Rig root directory containing images/ and sparse/0/.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Known-pose BA workspace. Default: <rig-dir>/known_pose_ba.",
    )
    parser.add_argument(
        "--input-model-dir",
        type=Path,
        default=None,
        help=(
            "Optional override for the previous optimized model dir. "
            "Defaults to the latest model resolved from <run-root>."
        ),
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help=(
            "Optional override for the working database path. By default, non-LTG runs use "
            "<run-root>/database.db as the source and prepare a working copy under the current iteration."
        ),
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional override for the rig image root. Default: <rig-dir>/images.",
    )
    parser.add_argument(
        "--image-list-path",
        type=Path,
        default=None,
        help="Optional override for image_list.txt. Default: <run-root>/image_list.txt.",
    )
    parser.add_argument(
        "--rerun-geometric-verifier",
        type=int,
        default=1,
        help="Whether to rerun geometric_verifier before point_triangulator.",
    )
    parser.add_argument(
        "--ltg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use existing SuperPoint + LightGlue artifacts instead of reusing COLMAP SIFT matches.",
    )
    parser.add_argument(
        "--ltg-hloc-python",
        type=str,
        default=None,
        help="Python executable used for HLOC commands. Default: current Python.",
    )
    parser.add_argument(
        "--ltg-work-dir",
        type=Path,
        default=None,
        help="Directory containing LTG pairs/features/matches. Default: <run-root>/ltg.",
    )
    parser.add_argument(
        "--ltg-build-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow HLOC feature extraction and matching when LTG artifacts are missing. "
            "Default: fail instead of re-detecting/re-matching."
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
        "--use-gpu",
        type=int,
        default=1,
        help="Whether COLMAP geometric verification should use GPU.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value, e.g. '0' or '1'.",
    )
    parser.add_argument(
        "--guided-matching",
        type=int,
        default=1,
        help="Whether geometric_verifier enables COLMAP FeatureMatching.guided_matching.",
    )
    parser.add_argument(
        "--rig-verification",
        type=int,
        default=1,
        help="Whether geometric_verifier enables COLMAP FeatureMatching.rig_verification.",
    )
    parser.add_argument(
        "--sift-max-ratio",
        type=float,
        default=0.9,
        help="COLMAP SiftMatching.max_ratio.",
    )
    parser.add_argument(
        "--sift-max-distance",
        type=float,
        default=0.8,
        help="COLMAP SiftMatching.max_distance.",
    )
    parser.add_argument(
        "--tri-min-angle",
        type=float,
        default=1.5,
        help="COLMAP Mapper.tri_min_angle.",
    )
    parser.add_argument(
        "--filter-max-reproj-error",
        type=float,
        default=2.0,
        help="COLMAP Mapper.filter_max_reproj_error.",
    )
    parser.add_argument(
        "--tri-merge-max-reproj-error",
        type=float,
        default=2.0,
        help="COLMAP Mapper.tri_merge_max_reproj_error.",
    )
    parser.add_argument(
        "--tri-complete-max-reproj-error",
        type=float,
        default=2.0,
        help="COLMAP Mapper.tri_complete_max_reproj_error.",
    )
    parser.add_argument(
        "--ba-use-gpu",
        type=int,
        default=None,
        help="Whether the post-triangulation bundle_adjuster should use GPU. Default: follow --use-gpu.",
    )
    parser.add_argument(
        "--ba-gpu-index",
        type=int,
        default=0,
        help="GPU index used by point_triangulator and bundle_adjuster within visible CUDA devices.",
    )
    parser.add_argument(
        "--ba-refine-rig-from-world",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow bundle_adjuster to refine rig_from_world poses. Enabled by default.",
    )
    parser.add_argument(
        "--ba-refine-sensor-from-rig",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow bundle_adjuster to refine sensor_from_rig extrinsics. Enabled by default.",
    )
    parser.add_argument(
        "--no-rig-constraint",
        action="store_true",
        help=(
            "Drop rigs/frames metadata before triangulation and bundle adjustment so every "
            "image extrinsic is optimized independently."
        ),
    )
    parser.add_argument(
        "--sync-latest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After this iteration, sync the latest text model into <rig-dir>/sparse/0.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync the latest resolved result to <rig-dir>/sparse/0, without running triangulation/BA.",
    )
    parser.add_argument(
        "--sync-source-model-dir",
        type=Path,
        default=None,
        help="Optional explicit sync source text model dir. Overrides latest-summary auto resolution.",
    )
    parser.add_argument(
        "--sync-points3d-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit points3D.txt used during sync. Default: prefer the latest "
            "iteration raw/0/points3D.txt, otherwise <rig-dir>/../raw/0/points3D.txt."
        ),
    )
    parser.add_argument(
        "--sync-target-model-dir",
        type=Path,
        default=None,
        help="Optional explicit sync target. Default: <rig-dir>/sparse/0.",
    )
    parser.add_argument(
        "--sync-backup-root",
        type=Path,
        default=None,
        help="Optional backup root. Default: <rig-dir>/sparse/backups.",
    )
    parser.add_argument(
        "--sync-no-backup",
        action="store_true",
        help="Do not back up the existing sync target before replacement.",
    )
    parser.add_argument(
        "--sync-dry-run",
        action="store_true",
        help="Resolve sync source/target without modifying files.",
    )
    return parser.parse_args()


def ensure_colmap() -> None:
    colmap_path = shutil.which("colmap")
    if colmap_path is None:
        raise FileNotFoundError("colmap executable not found in PATH.")
    result = subprocess.run(
        ["colmap", "help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    print(f"Using COLMAP: {colmap_path}", flush=True)
    first_line = result.stdout.splitlines()[0] if result.stdout else "COLMAP help"
    print(first_line, flush=True)


def run_command(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path | None = None,
) -> None:
    print("Running:", " ".join(cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, check=True, env=env)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)


def run_command_capture(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
) -> str:
    print("Running:", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    return result.stdout


def ensure_colmap_model_dir(model_dir: Path, stems: Sequence[str]) -> Path:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model dir: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)

    missing: list[str] = []
    for stem in stems:
        has_text = (model_dir / f"{stem}.txt").exists()
        has_binary = (model_dir / f"{stem}.bin").exists()
        if not has_text and not has_binary:
            missing.append(stem)
    if missing:
        raise FileNotFoundError(
            f"Incomplete COLMAP model in {model_dir}. Missing files for: {', '.join(missing)}"
        )
    return model_dir


def ensure_text_model_dir(model_dir: Path, filenames: Sequence[str]) -> Path:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing text model dir: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    missing = [filename for filename in filenames if not (model_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete text model in {model_dir}. Missing files: {', '.join(missing)}"
        )
    return model_dir


def model_dir_has_stem(model_dir: Path, stem: str) -> bool:
    return (model_dir / f"{stem}.txt").exists() or (model_dir / f"{stem}.bin").exists()


def resolve_model_stems(model_dir: Path) -> tuple[str, ...]:
    model_dir = ensure_colmap_model_dir(model_dir, STANDARD_MODEL_STEMS)
    has_rigs = model_dir_has_stem(model_dir, "rigs")
    has_frames = model_dir_has_stem(model_dir, "frames")
    if has_rigs != has_frames:
        raise FileNotFoundError(
            f"Incomplete rig metadata in {model_dir}. Expected both rigs.* and frames.*."
        )
    return RIG_MODEL_STEMS if has_rigs else STANDARD_MODEL_STEMS


def resolve_text_model_filenames(model_dir: Path) -> tuple[str, ...]:
    model_dir = ensure_text_model_dir(model_dir, STANDARD_SYNC_TEXT_FILES)
    has_rigs = (model_dir / "rigs.txt").is_file()
    has_frames = (model_dir / "frames.txt").is_file()
    if has_rigs != has_frames:
        raise FileNotFoundError(
            f"Incomplete rig text metadata in {model_dir}. Expected both rigs.txt and frames.txt."
        )
    return RIG_SYNC_TEXT_FILES if has_rigs else STANDARD_SYNC_TEXT_FILES


def copy_database_compatible(src: Path, dst: Path) -> None:
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        shutil.copyfile(src, dst)


def _table_columns(cur: sqlite3.Cursor, table_name: str, schema: str = "main") -> tuple[str, ...]:
    return tuple(
        str(row[1]) for row in cur.execute(f"PRAGMA {schema}.table_info({table_name})").fetchall()
    )


def inspect_cli_database_compatibility(
    database_path: Path,
    expected_image_names: Iterable[str] | None = None,
) -> tuple[bool, str]:
    if not database_path.exists():
        return False, f"database does not exist: {database_path}"

    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        table_names = {
            str(row[0])
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        required_tables = {
            "cameras",
            "images",
            "keypoints",
            "descriptors",
            "matches",
            "two_view_geometries",
            "pose_priors",
            "rigs",
            "rig_sensors",
            "frames",
            "frame_data",
        }
        missing_tables = sorted(required_tables - table_names)
        if missing_tables:
            return False, f"database is missing tables: {missing_tables}"

        descriptor_columns = _table_columns(cur, "descriptors")
        if descriptor_columns != CLI_DESCRIPTOR_COLUMNS:
            return False, f"descriptors schema is incompatible: {list(descriptor_columns)}"

        pose_prior_columns = _table_columns(cur, "pose_priors")
        if pose_prior_columns != CLI_POSE_PRIOR_COLUMNS:
            return False, f"pose_priors schema is incompatible: {list(pose_prior_columns)}"

        if expected_image_names is not None:
            expected_names = {str(name) for name in expected_image_names}
            db_names = {
                str(row[0])
                for row in cur.execute("SELECT name FROM images ORDER BY image_id").fetchall()
            }
            if db_names != expected_names:
                missing_names = sorted(expected_names - db_names)[:5]
                extra_names = sorted(db_names - expected_names)[:5]
                return False, (
                    "database image set mismatch: "
                    f"missing={missing_names}, extra={extra_names}"
                )

        return True, "database is CLI-compatible"
    finally:
        conn.close()


def rebuild_cli_database_from_source(
    source_database_path: Path,
    target_database_path: Path,
    env: dict[str, str],
) -> None:
    source_database_path = source_database_path.expanduser().resolve()
    target_database_path = target_database_path.expanduser().resolve()
    if not source_database_path.exists():
        raise FileNotFoundError(source_database_path)
    if target_database_path.exists():
        target_database_path.unlink()

    run_command(
        ["colmap", "database_creator", "--database_path", str(target_database_path)],
        env=env,
        log_path=None,
    )

    conn = sqlite3.connect(str(target_database_path))
    try:
        cur = conn.cursor()
        cur.execute("ATTACH DATABASE ? AS src", (str(source_database_path),))

        source_tables = {
            str(row[0])
            for row in cur.execute("SELECT name FROM src.sqlite_master WHERE type='table'").fetchall()
        }
        required_source_tables = {
            "cameras",
            "images",
            "keypoints",
            "descriptors",
            "matches",
            "two_view_geometries",
        }
        missing_source_tables = sorted(required_source_tables - source_tables)
        if missing_source_tables:
            raise RuntimeError(
                f"Source database is missing required tables: {missing_source_tables}"
            )

        cur.execute(
            """
            INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
            SELECT camera_id, model, width, height, params, prior_focal_length FROM src.cameras
            """
        )
        cur.execute(
            """
            INSERT INTO images(image_id, name, camera_id)
            SELECT image_id, name, camera_id FROM src.images
            """
        )
        cur.execute(
            """
            INSERT INTO keypoints(image_id, rows, cols, data)
            SELECT image_id, rows, cols, data FROM src.keypoints
            """
        )

        descriptor_columns = _table_columns(cur, "descriptors", schema="src")
        if descriptor_columns == CLI_DESCRIPTOR_COLUMNS or descriptor_columns[:4] == CLI_DESCRIPTOR_COLUMNS:
            cur.execute(
                """
                INSERT INTO descriptors(image_id, rows, cols, data)
                SELECT image_id, rows, cols, data FROM src.descriptors
                """
            )
        else:
            raise RuntimeError(
                "Cannot rebuild CLI database because source descriptors schema is unsupported: "
                f"{list(descriptor_columns)}"
            )

        cur.execute(
            """
            INSERT INTO matches(pair_id, rows, cols, data)
            SELECT pair_id, rows, cols, data FROM src.matches
            """
        )
        cur.execute(
            """
            INSERT INTO two_view_geometries(pair_id, rows, cols, data, config, F, E, H, qvec, tvec)
            SELECT pair_id, rows, cols, data, config, F, E, H, qvec, tvec
            FROM src.two_view_geometries
            """
        )

        if "pose_priors" in source_tables:
            pose_prior_columns = _table_columns(cur, "pose_priors", schema="src")
            if pose_prior_columns == CLI_POSE_PRIOR_COLUMNS:
                cur.execute(
                    """
                    INSERT INTO pose_priors(image_id, position, coordinate_system, position_covariance)
                    SELECT image_id, position, coordinate_system, position_covariance
                    FROM src.pose_priors
                    """
                )
            elif {
                "corr_data_id",
                "position",
                "coordinate_system",
                "position_covariance",
            }.issubset(set(pose_prior_columns)):
                cur.execute(
                    """
                    INSERT INTO pose_priors(image_id, position, coordinate_system, position_covariance)
                    SELECT corr_data_id, position, coordinate_system, position_covariance
                    FROM src.pose_priors
                    """
                )
            else:
                print(
                    "Skipping incompatible pose_priors during CLI database rebuild: "
                    f"{list(pose_prior_columns)}",
                    flush=True,
                )

        conn.commit()
        cur.execute("DETACH DATABASE src")
    finally:
        conn.close()


def prepare_working_database(
    source_database_path: Path,
    working_database_path: Path,
    expected_image_names: Iterable[str],
    env: dict[str, str],
) -> Path:
    source_database_path = source_database_path.expanduser().resolve()
    working_database_path = working_database_path.expanduser().resolve()

    compatible, reason = inspect_cli_database_compatibility(
        source_database_path,
        expected_image_names=expected_image_names,
    )
    if compatible:
        if source_database_path != working_database_path:
            working_database_path.parent.mkdir(parents=True, exist_ok=True)
            if working_database_path.exists():
                working_database_path.unlink()
            copy_database_compatible(source_database_path, working_database_path)
            print(
                f"Prepared working database copy from {source_database_path} to {working_database_path}",
                flush=True,
            )
            return working_database_path

        print(f"Using compatible database in place: {working_database_path}", flush=True)
        return working_database_path

    print(
        f"Rebuilding CLI-compatible database from {source_database_path}: {reason}",
        flush=True,
    )
    if source_database_path == working_database_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rebuilt_path = working_database_path.with_name(
            f".{working_database_path.stem}_cli_rebuilt_{timestamp}{working_database_path.suffix}"
        )
    else:
        rebuilt_path = working_database_path
        rebuilt_path.parent.mkdir(parents=True, exist_ok=True)
        if rebuilt_path.exists():
            rebuilt_path.unlink()

    rebuild_cli_database_from_source(
        source_database_path=source_database_path,
        target_database_path=rebuilt_path,
        env=env,
    )

    rebuilt_compatible, rebuilt_reason = inspect_cli_database_compatibility(
        rebuilt_path,
        expected_image_names=expected_image_names,
    )
    if not rebuilt_compatible:
        raise RuntimeError(
            "Rebuilt database is still incompatible with COLMAP CLI: "
            f"{rebuilt_reason}"
        )

    if source_database_path == working_database_path:
        source_database_path.unlink()
        rebuilt_path.rename(working_database_path)
        return working_database_path

    return rebuilt_path


def maybe_read_summary(summary_path: Path) -> dict[str, object] | None:
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def iteration_sort_key(iter_dir: Path) -> int:
    suffix = iter_dir.name.removeprefix("iter_")
    return int(suffix) if suffix.isdigit() else -1


def iter_summary_paths(run_root: Path) -> Iterable[Path]:
    latest_summary_path = run_root / "latest_summary.json"
    if latest_summary_path.exists():
        yield latest_summary_path

    iterations_root = run_root / "iterations"
    if iterations_root.exists():
        iter_dirs = sorted(
            (path for path in iterations_root.iterdir() if path.is_dir() and path.name.startswith("iter_")),
            key=iteration_sort_key,
            reverse=True,
        )
        for iter_dir in iter_dirs:
            summary_path = iter_dir / "summary.json"
            if summary_path.exists():
                yield summary_path

    base_summary_path = run_root / "summary.json"
    if base_summary_path.exists():
        yield base_summary_path


def resolve_input_model_dir(
    rig_dir: Path,
    run_root: Path,
    override: Path | None,
    require_rig_metadata: bool,
) -> Path:
    def ensure_candidate(candidate: Path) -> Path:
        candidate = candidate.expanduser().resolve()
        stems = resolve_model_stems(candidate)
        if require_rig_metadata and stems != RIG_MODEL_STEMS:
            raise FileNotFoundError(
                f"Model does not contain rig/frame metadata: {candidate}. "
                "Pass --no-rig-constraint if you want to optimize all image extrinsics independently."
            )
        return candidate

    if override is not None:
        return ensure_candidate(override)

    summary_keys = (
        "output_model_dir",
        "bundle_adjusted_model_dir",
        "refined_model_dir",
        "output_text_dir",
        "bundle_adjusted_text_dir",
        "refined_model_txt_dir",
        "published_sparse_dir",
    )
    for summary_path in iter_summary_paths(run_root):
        data = maybe_read_summary(summary_path)
        if data is None:
            continue
        for key in summary_keys:
            value = data.get(key)
            if not value:
                continue
            try:
                return ensure_candidate(Path(str(value)))
            except (FileNotFoundError, NotADirectoryError):
                continue

    sparse_zero = rig_dir / "sparse" / "0"
    if sparse_zero.exists():
        return ensure_candidate(sparse_zero)

    raise FileNotFoundError(
        f"Could not resolve a previous COLMAP model under {run_root}. "
        "Expected latest_summary.json, iterations/iter_*/summary.json, summary.json, or sparse/0."
    )


def resolve_sync_source_model_dir(run_root: Path, explicit_model_dir: Path | None) -> tuple[Path, str, Path | None]:
    if explicit_model_dir is not None:
        return (
            ensure_text_model_dir(explicit_model_dir, STANDARD_SYNC_TEXT_FILES),
            "explicit_sync_source_model_dir",
            None,
        )

    summary_keys = (
        "output_text_dir",
        "bundle_adjusted_text_dir",
        "refined_model_txt_dir",
    )
    for summary_path in iter_summary_paths(run_root):
        data = maybe_read_summary(summary_path)
        if data is None:
            continue
        for key in summary_keys:
            value = data.get(key)
            if not value:
                continue
            try:
                return (
                    ensure_text_model_dir(Path(str(value)), STANDARD_SYNC_TEXT_FILES),
                    key,
                    summary_path,
                )
            except (FileNotFoundError, NotADirectoryError):
                continue

    raise FileNotFoundError(
        "Could not resolve a sync source text model. Expected one of "
        "latest_summary.json, iterations/iter_*/summary.json, or summary.json."
    )


def resolve_sync_points3d_path(
    rig_dir: Path,
    run_root: Path,
    explicit_points3d_path: Path | None,
) -> tuple[Path, str, Path | None]:
    if explicit_points3d_path is not None:
        candidate = explicit_points3d_path.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing --sync-points3d-path: {candidate}")
        return candidate, "explicit_sync_points3d_path", None

    for summary_path in iter_summary_paths(run_root):
        data = maybe_read_summary(summary_path)
        if data is None:
            continue

        raw_text_dir = data.get("raw_text_dir")
        if raw_text_dir:
            candidate = Path(str(raw_text_dir)).expanduser().resolve() / "points3D.txt"
            if candidate.is_file():
                return candidate, "summary_raw_text_dir", summary_path

    default_candidate = (rig_dir.parent / "raw" / "0" / "points3D.txt").resolve()
    if default_candidate.is_file():
        return default_candidate, "default_workspace_raw", None

    raise FileNotFoundError(
        "Could not resolve sync points3D.txt. Expected latest raw_text_dir/points3D.txt "
        f"or {default_candidate}."
    )


def next_iteration_dir(run_root: Path) -> Path:
    iterations_root = run_root / "iterations"
    iterations_root.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    for path in iterations_root.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"iter_(\d+)", path.name)
        if match is not None:
            existing_ids.append(int(match.group(1)))
    next_id = max(existing_ids, default=0) + 1
    iteration_dir = iterations_root / f"iter_{next_id:04d}"
    iteration_dir.mkdir(parents=True, exist_ok=False)
    return iteration_dir


def sqlite_binary_params(params: np.ndarray) -> sqlite3.Binary:
    return sqlite3.Binary(np.asarray(params, dtype=np.float64).tobytes())


def sensor_sort_key(sensor: pycolmap.sensor_t) -> tuple[int, int]:
    return int(sensor.type), int(sensor.id)


def data_sort_key(data: pycolmap.data_t) -> tuple[int, int, int]:
    return int(data.sensor_id.type), int(data.sensor_id.id), int(data.id)


def load_reconstruction(model_dir: Path, require_rig_metadata: bool) -> pycolmap.Reconstruction:
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    if reconstruction.num_images() <= 0:
        raise RuntimeError(f"Empty reconstruction: {model_dir}")
    if reconstruction.num_cameras() <= 0:
        raise RuntimeError(f"Reconstruction has no cameras: {model_dir}")
    if require_rig_metadata and (reconstruction.num_rigs() <= 0 or reconstruction.num_frames() <= 0):
        raise RuntimeError(f"Model does not contain rig/frame metadata: {model_dir}")
    return reconstruction


def camera_center_from_image(image: pycolmap.Image) -> np.ndarray:
    cam_from_world = image.cam_from_world()
    world_from_cam = cam_from_world.inverse()
    return np.asarray(world_from_cam.todict()["translation"], dtype=np.float64)


def build_iterative_rig_images(reconstruction: pycolmap.Reconstruction) -> list[IterativeRigImage]:
    rig_images: list[IterativeRigImage] = []
    ordered_images = sorted(reconstruction.images.values(), key=lambda image: int(image.image_id))
    for image in ordered_images:
        rig_images.append(
            IterativeRigImage(
                name=str(image.name),
                mapped_camera_id=int(image.camera_id),
                center=camera_center_from_image(image),
            )
        )
    return rig_images


def initialize_database_cameras(
    database_path: Path,
    reconstruction: pycolmap.Reconstruction,
) -> None:
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = OFF")
        cur.execute("DELETE FROM cameras")
        for camera in sorted(reconstruction.cameras.values(), key=lambda item: int(item.camera_id)):
            cur.execute(
                """
                INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(camera.camera_id),
                    int(camera.model),
                    int(camera.width),
                    int(camera.height),
                    sqlite_binary_params(np.asarray(camera.params, dtype=np.float64)),
                    int(bool(camera.has_prior_focal_length)),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def initialize_database_images(
    database_path: Path,
    reconstruction: pycolmap.Reconstruction,
) -> None:
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = OFF")
        cur.execute("DELETE FROM images")
        for image in sorted(reconstruction.images.values(), key=lambda item: int(item.image_id)):
            cur.execute(
                """
                INSERT INTO images(image_id, name, camera_id)
                VALUES (?, ?, ?)
                """,
                (
                    int(image.image_id),
                    str(image.name),
                    int(image.camera_id),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def patch_database_from_reconstruction(
    database_path: Path,
    reconstruction: pycolmap.Reconstruction,
    include_rig_metadata: bool,
) -> None:
    images_by_name = {str(image.name): image for image in reconstruction.images.values()}

    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = OFF")

        db_rows = cur.execute("SELECT image_id, name FROM images ORDER BY image_id").fetchall()
        db_names = {str(name) for _, name in db_rows}
        model_names = set(images_by_name.keys())
        if db_names != model_names:
            missing = sorted(model_names - db_names)[:5]
            extra = sorted(db_names - model_names)[:5]
            raise RuntimeError(
                f"Database/model image mismatch. Missing in database: {missing}, extra in database: {extra}"
            )

        name_to_db_image_id = {str(name): int(image_id) for image_id, name in db_rows}
        for name, image in images_by_name.items():
            db_image_id = name_to_db_image_id[name]
            if db_image_id != int(image.image_id):
                raise RuntimeError(
                    f"Database/model image_id mismatch for {name}: db={db_image_id}, model={image.image_id}"
                )

        cur.execute("DELETE FROM cameras")
        for camera in sorted(reconstruction.cameras.values(), key=lambda item: int(item.camera_id)):
            cur.execute(
                """
                INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(camera.camera_id),
                    int(camera.model),
                    int(camera.width),
                    int(camera.height),
                    sqlite_binary_params(np.asarray(camera.params, dtype=np.float64)),
                    int(bool(camera.has_prior_focal_length)),
                ),
            )

        cur.execute("DELETE FROM frame_data")
        cur.execute("DELETE FROM frames")
        cur.execute("DELETE FROM rig_sensors")
        cur.execute("DELETE FROM rigs")

        for name, image in images_by_name.items():
            cur.execute(
                "UPDATE images SET camera_id = ? WHERE image_id = ?",
                (int(image.camera_id), name_to_db_image_id[name]),
            )

        if include_rig_metadata:
            if reconstruction.num_rigs() <= 0 or reconstruction.num_frames() <= 0:
                raise RuntimeError("Expected rig/frame metadata in the reconstruction.")

            for rig in sorted(reconstruction.rigs.values(), key=lambda item: int(item.rig_id)):
                ref_sensor = rig.ref_sensor_id
                cur.execute(
                    "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) VALUES (?, ?, ?)",
                    (
                        int(rig.rig_id),
                        int(ref_sensor.id),
                        int(ref_sensor.type),
                    ),
                )
                for sensor in sorted(rig.sensor_ids(), key=sensor_sort_key):
                    if rig.is_ref_sensor(sensor):
                        continue
                    cur.execute(
                        """
                        INSERT INTO rig_sensors(rig_id, sensor_id, sensor_type, sensor_from_rig)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            int(rig.rig_id),
                            int(sensor.id),
                            int(sensor.type),
                            None,
                        ),
                    )

            for frame in sorted(reconstruction.frames.values(), key=lambda item: int(item.frame_id)):
                cur.execute(
                    "INSERT INTO frames(frame_id, rig_id) VALUES (?, ?)",
                    (int(frame.frame_id), int(frame.rig_id)),
                )
                for data_id in sorted(frame.data_ids, key=data_sort_key):
                    cur.execute(
                        """
                        INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            int(frame.frame_id),
                            int(data_id.id),
                            int(data_id.sensor_id.id),
                            int(data_id.sensor_id.type),
                        ),
                    )

        conn.commit()
    finally:
        conn.close()


def prepare_input_model_without_rig_constraint(
    input_model_dir: Path,
    output_model_dir: Path,
    env: dict[str, str],
    log_path: Path,
) -> Path:
    output_model_dir.mkdir(parents=True, exist_ok=False)
    run_command(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(input_model_dir),
            "--output_path",
            str(output_model_dir),
            "--output_type",
            "TXT",
        ],
        env=env,
        log_path=log_path,
    )
    for filename in ("rigs.txt", "frames.txt"):
        file_path = output_model_dir / filename
        if file_path.exists():
            file_path.unlink()
    return ensure_text_model_dir(output_model_dir, STANDARD_SYNC_TEXT_FILES)


def summarize_database_matching(database_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        table_names = {
            str(row[0])
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

        def count_rows(table_name: str) -> int:
            if table_name not in table_names:
                return 0
            return int(cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])

        def sum_row_count_column(table_name: str) -> int:
            if table_name not in table_names:
                return 0
            columns = {
                str(row[1])
                for row in cur.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            if "rows" not in columns:
                return count_rows(table_name)
            value = cur.execute(
                f"SELECT COALESCE(SUM(rows), 0) FROM {table_name}"
            ).fetchone()[0]
            return int(value or 0)

        return {
            "num_images": count_rows("images"),
            "num_pose_priors": count_rows("pose_priors"),
            "num_matched_image_pairs": count_rows("matches"),
            "num_verified_image_pairs": count_rows("two_view_geometries"),
            "num_matches": sum_row_count_column("matches"),
            "num_inlier_matches": sum_row_count_column("two_view_geometries"),
        }
    finally:
        conn.close()


def count_points(points3d_path: Path) -> tuple[int, float]:
    num_points = 0
    total_track_length = 0
    with points3d_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            num_points += 1
            elems = stripped.split()
            total_track_length += max(0, (len(elems) - 8) // 2)
    mean_track_length = float(total_track_length / num_points) if num_points > 0 else 0.0
    return num_points, mean_track_length


def parse_model_analyzer_output(text: str) -> tuple[int | None, float | None]:
    num_observations = None
    mean_reprojection_error = None

    observations_match = re.search(r"Observations:\s+(\d+)", text)
    if observations_match is not None:
        num_observations = int(observations_match.group(1))

    reproj_match = re.search(r"Mean reprojection error:\s+([0-9eE+\-.]+)px", text)
    if reproj_match is not None:
        mean_reprojection_error = float(reproj_match.group(1))

    return num_observations, mean_reprojection_error


def run_ltg_matching(
    args: argparse.Namespace,
    database_path: Path,
    image_path: Path,
    rig_images: Sequence[IterativeRigImage],
    ltg_work_dir: Path,
    env: dict[str, str],
) -> object:
    ltg_utils_dir = Path(__file__).resolve().parent / "metacam_utils"
    if str(ltg_utils_dir) not in sys.path:
        sys.path.insert(0, str(ltg_utils_dir))

    from ltg_utils import (  # noqa: E402
        LtgPairingConfig,
        build_spatial_entries_from_rig_images,
        find_existing_ltg_artifacts,
        import_ltg_into_database,
        run_hloc_feature_pipeline,
    )

    entries = build_spatial_entries_from_rig_images(rig_images)
    pairing = LtgPairingConfig(
        local_window=int(args.ltg_local_window),
        global_every=int(args.ltg_global_every),
        spatial_num_neighbors=int(args.ltg_spatial_num_neighbors),
        spatial_max_distance=float(args.ltg_spatial_max_distance),
        netvlad_num_matched=int(args.ltg_netvlad_num_matched),
    )
    artifacts = find_existing_ltg_artifacts(
        ltg_work_dir,
        expected_image_names=(entry.name for entry in entries),
        expected_pairing=pairing,
    )
    if artifacts is None:
        if not bool(args.ltg_build_missing):
            raise FileNotFoundError(
                "LTG artifacts are missing or incompatible under "
                f"{ltg_work_dir}. Refusing to re-detect features/re-match by default. "
                "Reuse the known_pose_ba LTG directory or pass --ltg-build-missing to generate it."
            )
        artifacts = run_hloc_feature_pipeline(
            image_dir=image_path,
            work_dir=ltg_work_dir,
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


def copy_directory_compatible(source_dir: Path, target_dir: Path) -> None:
    source_dir = source_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if target_dir.exists():
        raise FileExistsError(f"Target directory already exists: {target_dir}")

    target_dir.mkdir(parents=True, exist_ok=False)
    for source_path in source_dir.iterdir():
        target_path = target_dir / source_path.name
        if source_path.is_symlink():
            resolved_source = source_path.resolve()
            if resolved_source.is_dir():
                copy_directory_compatible(resolved_source, target_path)
            else:
                copy_file_compatible(resolved_source, target_path)
            continue
        if source_path.is_dir():
            copy_directory_compatible(source_path, target_path)
            continue
        copy_file_compatible(source_path, target_path)


def resolve_backup_dir(backup_root: Path, target_model_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return backup_root / f"{target_model_dir.name}_before_replace_{timestamp}"


def sync_text_model_to_sparse(
    source_model_dir: Path,
    raw_points3d_path: Path,
    target_model_dir: Path,
    backup_root: Path | None,
    no_backup: bool,
    dry_run: bool,
) -> tuple[Path | None, bool]:
    source_model_dir = source_model_dir.expanduser().resolve()
    model_filenames = resolve_text_model_filenames(source_model_dir)
    raw_points3d_path = raw_points3d_path.expanduser().resolve()
    if not raw_points3d_path.is_file():
        raise FileNotFoundError(f"Missing sync points3D.txt: {raw_points3d_path}")

    target_model_dir = target_model_dir.expanduser().resolve()
    if target_model_dir.exists() and not target_model_dir.is_dir():
        raise NotADirectoryError(target_model_dir)

    print(f"Sync source model dir: {source_model_dir}", flush=True)
    print(f"Sync points3D path: {raw_points3d_path}", flush=True)
    print(f"Sync target model dir: {target_model_dir}", flush=True)
    if backup_root is not None:
        print(f"Sync backup root: {backup_root}", flush=True)
    else:
        print("Sync backup: disabled", flush=True)

    if dry_run:
        print("Sync dry run only, no files were changed.", flush=True)
        return None, False

    target_parent = target_model_dir.parent
    target_parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir = target_parent / f".{target_model_dir.name}_staging_{timestamp}"
    if staging_dir.exists():
        raise FileExistsError(f"Sync staging dir already exists: {staging_dir}")

    backup_dir = None
    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
        for filename in model_filenames:
            copy_file_compatible(source_model_dir / filename, staging_dir / filename)
        copy_file_compatible(raw_points3d_path, staging_dir / "points3D.txt")

        if target_model_dir.exists():
            if not no_backup:
                if backup_root is None:
                    raise ValueError("backup_root must not be None when backups are enabled.")
                backup_root.mkdir(parents=True, exist_ok=True)
                backup_dir = resolve_backup_dir(backup_root, target_model_dir)
                copy_directory_compatible(target_model_dir, backup_dir)
            shutil.rmtree(target_model_dir)

        staging_dir.rename(target_model_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    return backup_dir, True


def run_sync_only(args: argparse.Namespace, rig_dir: Path, run_root: Path) -> None:
    target_model_dir = (
        args.sync_target_model_dir.expanduser().resolve()
        if args.sync_target_model_dir is not None
        else (rig_dir / "sparse" / "0").resolve()
    )
    backup_root = (
        None
        if args.sync_no_backup
        else (
            args.sync_backup_root.expanduser().resolve()
            if args.sync_backup_root is not None
            else (rig_dir / "sparse" / "backups").resolve()
        )
    )

    source_model_dir, source_kind, source_summary_path = resolve_sync_source_model_dir(
        run_root=run_root,
        explicit_model_dir=args.sync_source_model_dir,
    )
    raw_points3d_path, points_kind, points_summary_path = resolve_sync_points3d_path(
        rig_dir=rig_dir,
        run_root=run_root,
        explicit_points3d_path=args.sync_points3d_path,
    )

    print(f"Resolved sync source kind: {source_kind}", flush=True)
    if source_summary_path is not None:
        print(f"Resolved source summary: {source_summary_path}", flush=True)
    print(f"Resolved sync points kind: {points_kind}", flush=True)
    if points_summary_path is not None:
        print(f"Resolved points summary: {points_summary_path}", flush=True)

    backup_dir, synced = sync_text_model_to_sparse(
        source_model_dir=source_model_dir,
        raw_points3d_path=raw_points3d_path,
        target_model_dir=target_model_dir,
        backup_root=backup_root,
        no_backup=bool(args.sync_no_backup),
        dry_run=bool(args.sync_dry_run),
    )
    if backup_dir is not None:
        print(f"Backed up previous sparse target to: {backup_dir}", flush=True)
    if synced:
        print(f"Synced latest text model to: {target_model_dir}", flush=True)


def main() -> None:
    args = parse_args()
    rig_dir = args.rig_dir.expanduser().resolve()
    run_root = (
        args.run_root.expanduser().resolve()
        if args.run_root is not None
        else (rig_dir / "known_pose_ba").resolve()
    )

    if not rig_dir.exists():
        raise FileNotFoundError(f"Missing rig dir: {rig_dir}")
    if not run_root.exists():
        raise FileNotFoundError(f"Missing run root: {run_root}")

    if args.sync_only:
        run_sync_only(args=args, rig_dir=rig_dir, run_root=run_root)
        return

    image_list_path = (
        args.image_list_path.expanduser().resolve()
        if args.image_list_path is not None
        else (run_root / "image_list.txt").resolve()
    )
    if not image_list_path.exists():
        raise FileNotFoundError(f"Missing image_list.txt: {image_list_path}")

    input_model_dir = resolve_input_model_dir(
        rig_dir=rig_dir,
        run_root=run_root,
        override=args.input_model_dir,
        require_rig_metadata=not bool(args.no_rig_constraint),
    )
    reconstruction = load_reconstruction(
        input_model_dir,
        require_rig_metadata=not bool(args.no_rig_constraint),
    )
    rig_images = build_iterative_rig_images(reconstruction)

    image_path_candidates = (
        [args.image_path.expanduser().resolve()]
        if args.image_path is not None
        else [rig_dir / "images"]
    )
    image_path = resolve_image_root_for_names(
        (str(image.name) for image in reconstruction.images.values()),
        image_path_candidates,
        label="rig image path",
    )

    ensure_colmap()

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"Using CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}", flush=True)

    ba_use_gpu = args.use_gpu if args.ba_use_gpu is None else args.ba_use_gpu

    iteration_dir = next_iteration_dir(run_root)
    raw_model_dir = iteration_dir / "raw_bin" / "0"
    raw_text_dir = iteration_dir / "raw" / "0"
    output_model_dir = iteration_dir / "sparse_bin" / "0"
    output_text_dir = iteration_dir / "sparse" / "0"
    raw_model_dir.mkdir(parents=True, exist_ok=False)
    raw_text_dir.mkdir(parents=True, exist_ok=False)
    output_model_dir.mkdir(parents=True, exist_ok=False)
    output_text_dir.mkdir(parents=True, exist_ok=False)

    point_triangulator_log_path = iteration_dir / "point_triangulator.log"
    bundle_adjuster_log_path = iteration_dir / "bundle_adjuster.log"
    geometric_verifier_log_path = (
        iteration_dir / "geometric_verifier.log" if int(args.rerun_geometric_verifier) else None
    )
    input_model_converter_log_path = iteration_dir / "input_model_converter.log"
    raw_model_converter_log_path = iteration_dir / "raw_model_converter.log"
    output_model_converter_log_path = iteration_dir / "output_model_converter.log"
    model_analyzer_log_path = iteration_dir / "model_analyzer.txt"
    summary_path = iteration_dir / "summary.json"

    source_database_path = (
        args.database_path.expanduser().resolve()
        if args.database_path is not None
        else (run_root / "database.db").resolve()
    )
    database_path = (
        args.database_path.expanduser().resolve()
        if args.database_path is not None
        else (
            (iteration_dir / "database_ltg.db").resolve()
            if args.ltg
            else (iteration_dir / "database.db").resolve()
        )
    )
    if not args.ltg and not source_database_path.exists():
        raise FileNotFoundError(source_database_path)
    ltg_work_dir = (
        args.ltg_work_dir.expanduser().resolve()
        if args.ltg_work_dir is not None
        else (run_root / "ltg").resolve()
    )

    ltg_artifacts = None
    if args.ltg:
        if int(args.rerun_geometric_verifier):
            print(
                "Ignoring --rerun-geometric-verifier because LTG imports and verifies matches directly.",
                flush=True,
            )
        if database_path.exists():
            database_path.unlink()
        run_command(
            ["colmap", "database_creator", "--database_path", str(database_path)],
            env=env,
            log_path=None,
        )
        initialize_database_cameras(database_path, reconstruction)
        initialize_database_images(database_path, reconstruction)
        ltg_artifacts = run_ltg_matching(
            args=args,
            database_path=database_path,
            image_path=image_path,
            rig_images=rig_images,
            ltg_work_dir=ltg_work_dir,
            env=env,
        )
    else:
        database_path = prepare_working_database(
            source_database_path=source_database_path,
            working_database_path=database_path,
            expected_image_names=(str(image.name) for image in reconstruction.images.values()),
            env=env,
        )

    input_model_has_rig_metadata = resolve_model_stems(input_model_dir) == RIG_MODEL_STEMS
    effective_input_model_dir = input_model_dir
    if args.no_rig_constraint:
        if input_model_has_rig_metadata:
            effective_input_model_dir = prepare_input_model_without_rig_constraint(
                input_model_dir=input_model_dir,
                output_model_dir=iteration_dir / "input_no_rig" / "0",
                env=env,
                log_path=input_model_converter_log_path,
            )
            print(
                "Prepared input model without rig constraints: "
                f"{effective_input_model_dir}",
                flush=True,
            )
        else:
            print(
                "Input model already has no rig/frame metadata; optimizing image extrinsics independently.",
                flush=True,
            )
        print(
            "Running without rig constraints: every image extrinsic will be optimized independently.",
            flush=True,
        )

    ba_refine_rig_from_world = bool(args.ba_refine_rig_from_world) and not bool(args.no_rig_constraint)
    ba_refine_sensor_from_rig = bool(args.ba_refine_sensor_from_rig) and not bool(args.no_rig_constraint)
    effective_rig_verification = bool(args.rig_verification) and not bool(args.no_rig_constraint)

    if bool(args.no_rig_constraint) and bool(args.rig_verification):
        print(
            "Disabling geometric_verifier rig_verification because --no-rig-constraint is enabled.",
            flush=True,
        )

    patch_database_from_reconstruction(
        database_path,
        reconstruction,
        include_rig_metadata=not bool(args.no_rig_constraint),
    )

    if int(args.rerun_geometric_verifier) and not args.ltg:
        run_command(
            [
                "colmap",
                "geometric_verifier",
                "--database_path",
                str(database_path),
                "--FeatureMatching.use_gpu",
                str(int(args.use_gpu)),
                "--FeatureMatching.guided_matching",
                str(int(args.guided_matching)),
                "--FeatureMatching.rig_verification",
                str(int(effective_rig_verification)),
                "--SiftMatching.max_ratio",
                str(float(args.sift_max_ratio)),
                "--SiftMatching.max_distance",
                str(float(args.sift_max_distance)),
            ],
            env=env,
            log_path=geometric_verifier_log_path,
        )

    matching_summary = summarize_database_matching(database_path)
    print(
        "Database matching summary: "
        f"{matching_summary['num_verified_image_pairs']} verified pairs, "
        f"{matching_summary['num_inlier_matches']} inlier matches",
        flush=True,
    )

    point_triangulator_cmd = [
        "colmap",
        "point_triangulator",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--input_path",
        str(effective_input_model_dir),
        "--output_path",
        str(raw_model_dir),
        "--refine_intrinsics",
        "0",
        "--Mapper.fix_existing_frames",
        "1",
        "--Mapper.tri_min_angle",
        str(float(args.tri_min_angle)),
        "--Mapper.filter_max_reproj_error",
        str(float(args.filter_max_reproj_error)),
        "--Mapper.tri_merge_max_reproj_error",
        str(float(args.tri_merge_max_reproj_error)),
        "--Mapper.tri_complete_max_reproj_error",
        str(float(args.tri_complete_max_reproj_error)),
        "--Mapper.ba_refine_focal_length",
        "0",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "0",
        "--Mapper.ba_refine_sensor_from_rig",
        "0",
        "--Mapper.ba_use_gpu",
        str(int(ba_use_gpu)),
        "--Mapper.image_list_path",
        str(image_list_path),
    ]
    if int(ba_use_gpu):
        point_triangulator_cmd.extend(
            [
                "--Mapper.ba_gpu_index",
                str(int(args.ba_gpu_index)),
            ]
        )
    run_command(point_triangulator_cmd, env=env, log_path=point_triangulator_log_path)

    run_command(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(raw_model_dir),
            "--output_path",
            str(raw_text_dir),
            "--output_type",
            "TXT",
        ],
        env=env,
        log_path=raw_model_converter_log_path,
    )

    bundle_adjuster_cmd = [
        "colmap",
        "bundle_adjuster",
        "--input_path",
        str(raw_model_dir),
        "--output_path",
        str(output_model_dir),
        "--BundleAdjustment.refine_focal_length",
        "0",
        "--BundleAdjustment.refine_principal_point",
        "0",
        "--BundleAdjustment.refine_extra_params",
        "0",
        "--BundleAdjustment.refine_rig_from_world",
        str(int(ba_refine_rig_from_world)),
        "--BundleAdjustment.refine_sensor_from_rig",
        str(int(ba_refine_sensor_from_rig)),
        "--BundleAdjustment.use_gpu",
        str(int(ba_use_gpu)),
    ]
    if int(ba_use_gpu):
        bundle_adjuster_cmd.extend(
            [
                "--BundleAdjustment.gpu_index",
                str(int(args.ba_gpu_index)),
            ]
        )
    run_command(bundle_adjuster_cmd, env=env, log_path=bundle_adjuster_log_path)

    run_command(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(output_model_dir),
            "--output_path",
            str(output_text_dir),
            "--output_type",
            "TXT",
        ],
        env=env,
        log_path=output_model_converter_log_path,
    )

    analyzer_output = run_command_capture(
        [
            "colmap",
            "model_analyzer",
            "--path",
            str(output_model_dir),
        ],
        env=env,
        log_path=model_analyzer_log_path,
    )
    num_observations, mean_reprojection_error = parse_model_analyzer_output(analyzer_output)

    raw_points3d_path = raw_text_dir / "points3D.txt"
    if not raw_points3d_path.exists():
        raise FileNotFoundError(f"Missing raw points3D.txt: {raw_points3d_path}")
    raw_num_points3d, raw_mean_track_length = count_points(raw_points3d_path)

    output_points3d_path = output_text_dir / "points3D.txt"
    if not output_points3d_path.exists():
        raise FileNotFoundError(f"Missing bundle-adjusted points3D.txt: {output_points3d_path}")
    output_num_points3d, output_mean_track_length = count_points(output_points3d_path)

    sync_target_model_dir = (
        args.sync_target_model_dir.expanduser().resolve()
        if args.sync_target_model_dir is not None
        else (rig_dir / "sparse" / "0").resolve()
    )
    sync_backup_root = (
        None
        if args.sync_no_backup
        else (
            args.sync_backup_root.expanduser().resolve()
            if args.sync_backup_root is not None
            else (rig_dir / "sparse" / "backups").resolve()
        )
    )
    sync_backup_dir = None
    synced_latest = False
    if args.sync_latest:
        sync_points3d_path = (
            args.sync_points3d_path.expanduser().resolve()
            if args.sync_points3d_path is not None
            else raw_points3d_path
        )
        sync_backup_dir, synced_latest = sync_text_model_to_sparse(
            source_model_dir=output_text_dir,
            raw_points3d_path=sync_points3d_path,
            target_model_dir=sync_target_model_dir,
            backup_root=sync_backup_root,
            no_backup=bool(args.sync_no_backup),
            dry_run=bool(args.sync_dry_run),
        )
        if sync_backup_dir is not None:
            print(f"Backed up previous sparse target to: {sync_backup_dir}", flush=True)
        if synced_latest:
            print(f"Synced latest text model to: {sync_target_model_dir}", flush=True)
    else:
        sync_points3d_path = (
            args.sync_points3d_path.expanduser().resolve()
            if args.sync_points3d_path is not None
            else None
        )

    summary = IterativeRigBaSummary(
        rig_dir=str(rig_dir),
        run_root=str(run_root),
        iteration_dir=str(iteration_dir),
        database_path=str(database_path),
        input_model_dir=str(input_model_dir),
        raw_model_dir=str(raw_model_dir),
        raw_text_dir=str(raw_text_dir),
        output_model_dir=str(output_model_dir),
        output_text_dir=str(output_text_dir),
        no_rig_constraint=bool(args.no_rig_constraint),
        rerun_geometric_verifier=bool(args.rerun_geometric_verifier) and not bool(args.ltg),
        ltg=bool(args.ltg),
        guided_matching=bool(args.guided_matching),
        rig_verification=bool(effective_rig_verification),
        triangulator_fix_existing_frames=True,
        bundle_adjust_refine_rig_from_world=bool(ba_refine_rig_from_world),
        bundle_adjust_refine_sensor_from_rig=bool(ba_refine_sensor_from_rig),
        bundle_adjust_refine_intrinsics=False,
        ltg_work_dir=(str(ltg_artifacts.work_dir) if ltg_artifacts is not None else None),
        ltg_pairs_path=(str(ltg_artifacts.pairs_path) if ltg_artifacts is not None else None),
        ltg_features_path=(str(ltg_artifacts.features_path) if ltg_artifacts is not None else None),
        ltg_matches_path=(str(ltg_artifacts.matches_path) if ltg_artifacts is not None else None),
        ltg_pair_count=(int(ltg_artifacts.num_pairs) if ltg_artifacts is not None else None),
        point_triangulator_log_path=str(point_triangulator_log_path),
        bundle_adjuster_log_path=str(bundle_adjuster_log_path),
        geometric_verifier_log_path=(
            str(geometric_verifier_log_path) if geometric_verifier_log_path is not None else None
        ),
        raw_model_converter_log_path=str(raw_model_converter_log_path),
        output_model_converter_log_path=str(output_model_converter_log_path),
        model_analyzer_log_path=str(model_analyzer_log_path),
        num_images=int(reconstruction.num_images()),
        num_cameras=int(reconstruction.num_cameras()),
        raw_num_points3d=raw_num_points3d,
        raw_mean_track_length=raw_mean_track_length,
        output_num_points3d=output_num_points3d,
        output_mean_track_length=output_mean_track_length,
        num_observations=num_observations,
        mean_reprojection_error=mean_reprojection_error,
        sync_latest_requested=bool(args.sync_latest),
        sync_dry_run=bool(args.sync_dry_run),
        synced_latest=bool(synced_latest),
        sync_target_model_dir=(str(sync_target_model_dir) if args.sync_latest else None),
        sync_points3d_path=(str(sync_points3d_path) if sync_points3d_path is not None else None),
        sync_backup_dir=(str(sync_backup_dir) if sync_backup_dir is not None else None),
    )

    summary_dict = asdict(summary)
    summary_dict["matching_database"] = matching_summary
    summary_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    (run_root / "latest_summary.json").write_text(
        json.dumps(summary_dict, indent=2),
        encoding="utf-8",
    )

    print(f"Iteration dir: {iteration_dir}", flush=True)
    print(f"Previous model dir: {input_model_dir}", flush=True)
    if effective_input_model_dir != input_model_dir:
        print(f"Effective input model dir: {effective_input_model_dir}", flush=True)
    print(f"Raw model dir: {raw_model_dir}", flush=True)
    print(f"Raw text model dir: {raw_text_dir}", flush=True)
    print(f"New model dir: {output_model_dir}", flush=True)
    print(f"New text model dir: {output_text_dir}", flush=True)
    print(f"Point triangulator log: {point_triangulator_log_path}", flush=True)
    print(f"Bundle adjuster log: {bundle_adjuster_log_path}", flush=True)
    if geometric_verifier_log_path is not None:
        print(f"Geometric verifier log: {geometric_verifier_log_path}", flush=True)
    print(f"Model analyzer log: {model_analyzer_log_path}", flush=True)
    print(f"Raw points3D.txt: {raw_points3d_path}", flush=True)
    print(f"Raw points: {raw_num_points3d}", flush=True)
    print(f"Bundle-adjusted points: {output_num_points3d}", flush=True)
    print(f"Raw mean track length: {raw_mean_track_length:.3f}", flush=True)
    print(f"Bundle-adjusted mean track length: {output_mean_track_length:.3f}", flush=True)
    print(f"Observations: {num_observations}", flush=True)
    if mean_reprojection_error is not None:
        print(f"Mean reprojection error: {mean_reprojection_error:.6f}px", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
