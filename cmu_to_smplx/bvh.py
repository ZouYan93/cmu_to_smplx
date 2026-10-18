"""Strict cgspeed reader using SciPy intrinsic ZYX rotations."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from .conventions import require, validate_hierarchy


@dataclass
class Animation:
    bones: tuple[str, ...]
    parents: np.ndarray
    offsets: np.ndarray
    pos: np.ndarray
    quats: np.ndarray  # Local wxyz, including the converter calibration row.
    frame_time: float


def _header(text):
    tokens = iter(re.findall(r"[{}]|[^\s{}]+", text))
    names, parents, offsets = [], [], []

    def take(expected=None):
        token = next(tokens)
        require(expected is None or token == expected,
                "Unexpected BVH hierarchy token: " + token)
        return token

    def offset():
        take("OFFSET")
        value = [float(take()) for _ in range(3)]
        require(np.isfinite(value).all(), "Nonfinite BVH offset")
        return value

    def joint(parent):
        name = take()
        require(re.fullmatch(r"\w+", name) is not None and len(names) < 31,
                "Unsupported BVH joint name or joint count")
        index = len(names)
        names.append(name)
        parents.append(parent)
        take("{")
        offsets.append(offset())
        take("CHANNELS")
        expected = (["6", "Xposition", "Yposition", "Zposition"] if parent < 0 else ["3"])
        expected += ["Zrotation", "Yrotation", "Xrotation"]
        require([take() for _ in expected] == expected,
                "Expected cgspeed root XYZ position and intrinsic ZYX rotations")
        token = take()
        while token in ("JOINT", "End"):
            if token == "JOINT":
                joint(index)
            else:
                take("Site")
                take("{")
                offset()
                take("}")
            token = take()
        require(token == "}", "Unclosed BVH joint")

    try:
        take("HIERARCHY")
        take("ROOT")
        joint(-1)
        require(next(tokens, None) is None, "Trailing BVH hierarchy tokens")
    except (StopIteration, RecursionError) as exc:
        raise ValueError("Incomplete BVH hierarchy") from exc
    parents = np.asarray(parents, dtype=np.int64)
    validate_hierarchy(names, parents)
    return tuple(names), parents, np.asarray(offsets, dtype=np.float64)


def read_bvh(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 128 * 1024**2,
            "BVH is missing or exceeds128MiB")
    with path.open(encoding="utf-8-sig") as stream:
        lines = []
        for _ in range(4097):
            line = stream.readline()
            require(bool(line), "BVH has no MOTION section")
            if line.strip() == "MOTION":
                break
            lines.append(line)
        else:
            raise ValueError("BVH hierarchy exceeds4096 lines")
        names, parents, offsets = _header("".join(lines))
        count = re.fullmatch(r"Frames:\s*(\d+)", stream.readline().strip())
        step = re.fullmatch(r"Frame Time:\s*([0-9.eE+-]+)", stream.readline().strip())
        require(count is not None and step is not None, "Invalid BVH motion header")
        count, step = int(count[1]), float(step[1])
        require(3 <= count <= 72001 and np.isfinite(step) and step > 0,
                "BVH needs one calibration row and2 to72000 action rows")
        values = np.loadtxt(stream, dtype=np.float64, ndmin=2, comments=None)
    require(values.shape == (count, 96) and np.isfinite(values).all(),
            "BVH motion row count, channel count or finite values differ")
    positions = np.broadcast_to(offsets, (count, 31, 3)).copy()
    # The converter's root channels replace its root OFFSET.
    positions[:, 0] = values[:, :3]
    angles = values[:, 3:].reshape(count, 31, 3)
    quats = Rotation.from_euler("ZYX", angles.reshape(-1, 3), degrees=True)
    return Animation(names, parents, offsets, positions,
                     quats.as_quat(scalar_first=True).reshape(count, 31, 4), step)


def has_calibration_row(anim):
    """Retain the Worker's 2008/2010 converter signature and tolerances."""
    index = {name: j for j, name in enumerate(anim.bones)}
    q = anim.quats[0]

    def z_angle(name):
        value = q[index[name]]
        return float(2 * np.arctan2(abs(value[3]), max(abs(value[0]), 1e-8)))

    return bool(
        all(.20 <= z_angle(name) <= .45 for name in ("LeftUpLeg", "RightUpLeg"))
        and all(.04 <= z_angle(name) <= .20 for name in ("LeftArm", "RightArm"))
        and q[index["LeftUpLeg"], 3] * q[index["RightUpLeg"], 3] < 0
        and q[index["LeftArm"], 3] * q[index["RightArm"], 3] < 0
        and np.linalg.norm(q[index["Spine1"], 1:]) < .02
    )


def forward_kinematics(anim):
    count = len(anim.quats)
    local = Rotation.from_quat(anim.quats.reshape(-1, 4), scalar_first=True)
    matrices = local.as_matrix().reshape(count, 31, 3, 3)
    world = matrices.copy()
    points = anim.pos.copy()
    for j, parent in enumerate(anim.parents[1:], 1):
        world[:, j] = world[:, parent] @ matrices[:, j]
        points[:, j] = (points[:, parent]
                        + np.einsum("fij,fj->fi", world[:, parent], anim.pos[:, j]))
    return world, points


def rest_positions(offsets, parents):
    result = np.asarray(offsets, dtype=np.float64).copy()
    for j, parent in enumerate(parents):
        if parent >= 0:
            result[j] += result[parent]
    return result
