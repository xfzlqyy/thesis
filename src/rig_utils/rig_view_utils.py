"""Shared rig-view helpers aligned with colmap_360_to_rigs.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_RIG_FOV = 60.0


@dataclass(frozen=True)
class ViewSpec:
    name: str
    rotation: np.ndarray


def default_rig_rotations() -> Dict[str, np.ndarray]:
    return {
        "front": np.eye(3, dtype=np.float64),
        "left": cv2.Rodrigues(np.array([0.0, np.pi / 4.0, 0.0], dtype=np.float64))[0],
        "right": cv2.Rodrigues(np.array([0.0, -np.pi / 4.0, 0.0], dtype=np.float64))[0],
        "top": cv2.Rodrigues(np.array([-np.pi / 4.0, 0.0, 0.0], dtype=np.float64))[0],
        "bottom": cv2.Rodrigues(np.array([np.pi / 4.0, 0.0, 0.0], dtype=np.float64))[0],
    }


def default_view_specs() -> List[ViewSpec]:
    return [
        ViewSpec(name=name, rotation=rotation.copy())
        for name, rotation in default_rig_rotations().items()
    ]


def build_perspective_k(output_size: Tuple[int, int], fov_degrees: float) -> np.ndarray:
    width, height = output_size
    focal = width / (2.0 * np.tan(np.radians(fov_degrees / 2.0)))
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Legacy view spec parser kept for CLI compatibility."""
    r_yaw = cv2.Rodrigues(np.array([0.0, np.radians(yaw_deg), 0.0], dtype=np.float64))[0]
    r_pitch = cv2.Rodrigues(np.array([np.radians(-pitch_deg), 0.0, 0.0], dtype=np.float64))[0]
    return r_pitch @ r_yaw


def rotation_from_rotvec_deg(rotvec_deg: Sequence[float]) -> np.ndarray:
    rotvec_rad = np.radians(np.asarray(rotvec_deg, dtype=np.float64))
    return cv2.Rodrigues(rotvec_rad)[0]


def _parse_floats(spec: str, expected_count: int, view_def: str) -> List[float]:
    values = [item.strip() for item in spec.split(",")]
    if len(values) != expected_count:
        raise ValueError(f"Invalid view definition: {view_def}")
    return [float(item) for item in values]


def parse_view_definition(view_def: str) -> ViewSpec:
    view_def = view_def.strip()
    if "@rotvec_deg:" in view_def:
        name, raw_rotvec = view_def.split("@rotvec_deg:", 1)
        rotation = rotation_from_rotvec_deg(_parse_floats(raw_rotvec, 3, view_def))
    elif "@rotvec:" in view_def:
        name, raw_rotvec = view_def.split("@rotvec:", 1)
        rotation = cv2.Rodrigues(np.asarray(_parse_floats(raw_rotvec, 3, view_def), dtype=np.float64))[0]
    elif ":" in view_def:
        name, raw_angles = view_def.split(":", 1)
        yaw_deg, pitch_deg = _parse_floats(raw_angles, 2, view_def)
        rotation = rotation_from_yaw_pitch(yaw_deg, pitch_deg)
    else:
        raise ValueError(f"Invalid view definition: {view_def}")

    name = name.strip()
    if not name:
        raise ValueError(f"Invalid view definition: {view_def}")
    return ViewSpec(name=name, rotation=rotation)


def parse_view_definitions(view_defs: Iterable[str] | None) -> List[ViewSpec]:
    if not view_defs:
        return default_view_specs()
    return [parse_view_definition(view_def) for view_def in view_defs]
