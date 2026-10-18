"""Full native-frame conversion. No fitting, repair, smoothing or sampling."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .asf import restore_geometry
from .bvh import forward_kinematics, has_calibration_row, read_bvh, rest_positions
from .conventions import BODY_NAMES, METHOD, POSITION_SOURCES, ROTATION_SOURCES, WORLD, require
from .mapping import build_offsets, calibration, file_sha256, map_rotations, model_rest
from .timing import resolve_timing


@dataclass
class Conversion:
    root_orient: np.ndarray
    pose_body: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    mocap_frame_rate: float
    provenance: dict

    def arrays(self):
        return dict(root_orient=self.root_orient, pose_body=self.pose_body,
                    trans=self.trans, betas=self.betas,
                    mocap_frame_rate=np.asarray(self.mocap_frame_rate, dtype=np.float64),
                    up_axis=np.asarray("Z"), gender=np.asarray("neutral"))


def convert(bvh, asf, model, *, fps=None, timing_manifest=None, sequence_id=None):
    """Map one complete original cgspeed BVH with its matching official ASF."""
    paths = dict(bvh=Path(bvh).resolve(), asf=Path(asf).resolve(), model=Path(model).resolve())
    if timing_manifest is not None:
        paths["timingManifest"] = Path(timing_manifest).resolve()
    pins = {name: file_sha256(path) for name, path in paths.items()}
    anim = read_bvh(paths["bvh"])
    require(has_calibration_row(anim), "Original prepended cgspeed calibration row is required")
    clock = resolve_timing(paths["bvh"], len(anim.quats), anim.frame_time,
                           fps=fps, manifest=timing_manifest, sequence_id=sequence_id)
    geometry = restore_geometry(anim, paths["asf"])
    rest, model_hash = model_rest(paths["model"])
    corrections, config, calibration_hash = calibration()
    world, positions = forward_kinematics(anim)
    canonical = WORLD @ world[0]
    source_rest = (rest_positions(anim.offsets, anim.parents) @ WORLD.T) * geometry["metersPerUnit"]
    offsets = build_offsets(anim.bones, anim.parents, source_rest, canonical, rest)
    ids = [anim.bones.index(name) for name in ROTATION_SOURCES]
    source_global = WORLD @ world[1:, ids]
    pose = map_rotations(source_global, offsets, corrections)
    # Preserve the original Hips path with the Worker's unrotated rest-pelvis subtraction.
    hips = (positions[1:, 0] @ WORLD.T) * geometry["metersPerUnit"]
    trans = hips - rest[0]
    require(all(file_sha256(path) == pins[name] for name, path in paths.items()),
            "Input changed during conversion")
    provenance = dict(
        schemaVersion="cmu-to-smplx-provenance/1", method=METHOD, packageVersion="0.1.0",
        inputs={name: dict(file=path.name, sha256=pins[name]) for name, path in paths.items()},
        modelSha256=model_hash, calibrationSha256=calibration_hash,
        originalWorkerCalibrationSha256=config["sourceConfigSha256"],
        retainedCalibrationSha256=config["retainedModelSha256"],
        offlineCalibration=config["provenance"], timing=clock, geometry=geometry,
        coordinates=dict(up="Z", handedness="right", lengthUnit="m", angleUnit="rad",
                         sourceToOutput="positive90 degrees about X"),
        rotationSources=list(ROTATION_SOURCES), positionSources=list(POSITION_SOURCES),
        targetJointNames=list(BODY_NAMES), targetBetas=[0.] * 16,
        rootPolicy="original Hips minus unrotated zero16 model rest pelvis",
        nativeFrameIdRange=[0, len(pose) - 1], nativeFrames=len(pose),
        nativeFps=clock["nativeFps"], durationSeconds=len(pose) / clock["nativeFps"],
        lastSampleSeconds=(len(pose) - 1) / clock["nativeFps"],
        smoothing=False, sourceRepair=False, resampled=False, perActionFitting=False,
        optimizationAtInference=False, targetMotionRead=False, targetShapeRead=False,
        handsAndFaceEstimated=False, anatomicalGroundTruth=False, productionApproved=False,
        qualityScreening="structural and numeric only; native rotation jumps are preserved",
    )
    return Conversion(pose[:, 0], pose[:, 1:].reshape(-1, 63), trans,
                      np.zeros(16), clock["nativeFps"], provenance)


def save(result, output):
    """Write an NPZ and matching JSON sidecar; existing files are preserved."""
    output = Path(output).resolve()
    require(output.suffix.lower() == ".npz", "Output path must end in .npz")
    sidecar = output.with_suffix(".json")
    require(not output.exists() and not sidecar.exists(), "Output NPZ or JSON already exists")
    encoded = json.dumps(result.provenance, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open("xb") as stream:
            created.append(output)
            np.savez_compressed(stream, **result.arrays())
        with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
            created.append(sidecar)
            stream.write(encoded)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return output, sidecar
