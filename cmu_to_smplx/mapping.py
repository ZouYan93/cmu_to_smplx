"""Static geometry plus the retained22 offline-learned corrections."""

from importlib.resources import files
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .conventions import (
    LIMB_EDGES, METHOD, MODEL_SHA256, PARENTS, POSITION_SOURCES,
    ROTATION_SOURCES, SOURCE_CONFIG_SHA256, WORLD, require, validate_hierarchy,
)

CALIBRATION_SHA256 = "d0155fd7cdfec17b6f30016237a15ada1a46afcd7d2d58aab16a945ca9c13d09"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proper(value):
    a = np.asarray(value, dtype=np.float64)
    require(a.shape[-2:] == (3, 3) and np.isfinite(a).all()
            and np.allclose(a.swapaxes(-1, -2) @ a, np.eye(3), atol=1e-8, rtol=0)
            and np.allclose(np.linalg.det(a), 1, atol=1e-8, rtol=0),
            "Expected proper finite rotation matrices")
    return a


def calibration():
    raw = files("cmu_to_smplx").joinpath("calibration.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == CALIBRATION_SHA256, "Retained calibration SHA256 differs")
    data = json.loads(raw)
    require(data.get("schemaVersion") == "cmu-standalone-calibration/1"
            and data.get("method") == METHOD
            and data.get("sourceConfigSha256") == SOURCE_CONFIG_SHA256
            and data.get("neutralModelSha256") == MODEL_SHA256
            and data.get("targetBetas") == [0.] * 16,
            "Retained calibration metadata differs")
    rotations = proper(data["rotations"])
    require(rotations.shape == (22, 3, 3), "Calibration needs22 shared rotations")
    return rotations, data, digest


def model_rest(path):
    """Regress rest joints at zero16 shape from an operator's licensed NPZ."""
    path = Path(path)
    digest = file_sha256(path)
    require(digest == MODEL_SHA256, "Neutral SMPL-X model SHA256 differs from retained calibration")
    with np.load(path, allow_pickle=False) as data:
        require(tuple(data["kintree_table"][0, 1:22]) == PARENTS[1:],
                "SMPL-X body hierarchy differs")
        rest = data["J_regressor"][:22].astype(float) @ data["v_template"].astype(float)
    require(rest.shape == (22, 3) and np.isfinite(rest).all(), "Invalid neutral model rest joints")
    return rest, digest


def swing(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    require(np.isfinite([na, nb]).all() and min(na, nb) > 1e-10,
            "Degenerate rest direction")
    a, b = a / na, b / nb
    c = float(np.clip(a @ b, -1., 1.))
    require(c > -1 + 1e-8, "Antiparallel rest direction has ambiguous swing")
    x, y, z = np.cross(a, b)
    k = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    return proper(np.eye(3) + k + k @ k / (1 + c))


def build_offsets(names, parents, source_rest, canonical, target_rest):
    validate_hierarchy(names, parents)
    rest, target = np.asarray(source_rest), np.asarray(target_rest)
    canonical = proper(canonical)
    require(rest.shape == (31, 3) and target.shape == (22, 3)
            and canonical.shape == (31, 3, 3)
            and np.isfinite(rest).all() and np.isfinite(target).all(),
            "Static geometry dimensions differ")
    points = np.zeros_like(rest)
    for j, parent in enumerate(parents[1:], 1):
        points[j] = points[parent] + canonical[parent] @ (WORLD.T @ (rest[j] - rest[parent]))
    frames = np.broadcast_to(WORLD, (22, 3, 3)).copy()
    for parent, child in LIMB_EDGES:
        a, b = (names.index(POSITION_SOURCES[j]) for j in (parent, child))
        frames[parent] = swing(WORLD @ (target[child] - target[parent]),
                               points[b] - points[a]) @ WORLD
    ids = [names.index(name) for name in ROTATION_SOURCES]
    return proper(canonical[ids].swapaxes(-1, -2) @ frames)


def map_rotations(source_global, offsets, corrections):
    source, offsets, corrections = proper(source_global), proper(offsets), proper(corrections)
    require(source.ndim == 4 and source.shape[1:] == (22, 3, 3)
            and offsets.shape == corrections.shape == (22, 3, 3),
            "Rotation mapping dimensions differ")
    world = source @ offsets @ corrections
    local = world.copy()
    local[:, 1:] = world[:, PARENTS[1:]].swapaxes(-1, -2) @ world[:, 1:]
    return Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 22, 3)


def body_fk(pose, rest):
    pose, rest = np.asarray(pose), np.asarray(rest)
    require(pose.ndim == 3 and pose.shape[1:] == (22, 3) and rest.shape == (22, 3)
            and np.isfinite(pose).all() and np.isfinite(rest).all(), "Invalid body FK input")
    world = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
    points = np.zeros((len(pose), 22, 3))
    points[:, 0] = rest[0]
    for j, parent in enumerate(PARENTS[1:], 1):
        world[:, j] = world[:, parent] @ world[:, j]
        points[:, j] = (points[:, parent]
                        + np.einsum("fij,j->fi", world[:, parent], rest[j] - rest[parent]))
    return points, world
