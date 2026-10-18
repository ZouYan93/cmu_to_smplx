"""Opt-in full-native parity with hash-frozen Worker pure prediction.

All captures, ASF and models remain external. Only scalar errors and hashes
enter the report. No Worker package is imported by cmu_to_smplx.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation

# Running this file directly keeps the local package importable without install.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cmu_to_smplx import convert, save
from cmu_to_smplx.bvh import forward_kinematics, read_bvh, rest_positions
from cmu_to_smplx.asf import restore_geometry
from cmu_to_smplx.conventions import WORLD
from cmu_to_smplx.mapping import (
    body_fk, build_offsets, calibration, file_sha256, model_rest,
)

FROZEN = {
    "worker/cmu_lightweight.py": "ef3e7af8e15316e2800613268abb0ce138d94dc373720c8ea0fbf1a82ec99620",
    "worker/cmu_lightweight.json": "6a20ea13a43b26400988eccfef4fa4bacebdd985713ffd863722526ec3986152",
    "worker/cmu_asf.py": "ff4c03b672c3b85b68be9a96c26c2fe52fe47f63b32a4e1c8a4f678525363f59",
    "worker/cmu_fixed_basis.py": "f7e1eaca7f457b3fc5662e152b75b09cc2312c4fa36c6de9476f1518c4153d7b",
    "worker/source_adapter.py": "1d530a87e37a183bffcab2c16ac4c6646addc69138f477661c97aba3e13f69f3",
    "worker/cmu_timing.py": "7a3d30a14028e25d8b816f7b22dc12b29650ef0d25d9a711c67e1d3c2183d155",
    "vendor/GMR/general_motion_retargeting/utils/lafan_vendor/extract.py":
        "fa3ae83d68d0a8239df51750bca179dd69dbcbbb2110ed63c3be70ef2273aa18",
    "vendor/GMR/general_motion_retargeting/utils/lafan_vendor/utils.py":
        "5f666f15f42cb82f7bb403e8fad33bd77b863f6d51af0eeee09908e9985ab50b",
}
IDS = ("02_01", "02_05", "06_14", "35_01")


def extract(path, names, namespace):
    module = ast.parse(path.read_bytes(), filename=str(path))
    selected = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
                selected.append(node)
    found = {node.name for node in selected if isinstance(node, ast.FunctionDef)}
    found.update(target.id for node in selected if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name))
    if not names <= found:
        raise ValueError("Missing frozen reference names: " + str(names - found))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference(worker):
    # Read all frozen files before evaluation; imports below never touch live services.
    for relative, expected in FROZEN.items():
        if file_sha256(worker / relative) != expected:
            raise ValueError("Frozen reference changed: " + relative)
    sys.dont_write_bytecode = True
    vendor_path = worker / "vendor/GMR/general_motion_retargeting/utils/lafan_vendor"
    package = types.ModuleType("_cmu_parity_vendor")
    package.__path__ = [str(vendor_path)]
    sys.modules[package.__name__] = package
    utils = load_module(package.__name__ + ".utils", vendor_path / "utils.py")
    reader = load_module(package.__name__ + ".extract", vendor_path / "extract.py")
    asf = load_module("_cmu_parity_asf", worker / "worker/cmu_asf.py")
    ns = dict(np=np, Rotation=Rotation, Path=Path)
    extract(worker / "worker/motion_filter.py", {"PARENTS"}, ns)
    extract(worker / "worker/cmu_fixed_basis.py", {"SOURCE_PARENTS", "validate_bvh_header"}, ns)
    extract(worker / "worker/source_adapter.py",
            {"_name_index", "_looks_like_cmu_converter_tpose", "hierarchy_rest_positions"}, ns)
    extract(worker / "worker/cmu_lightweight.py", {
        "METHOD", "WORLD", "ROTATION_SOURCES", "POSITION_SOURCES", "LIMB_EDGES", "EXPECTED_PARENTS",
        "_require", "_immutable", "_proper", "_hierarchy", "_swing",
        "build_profile", "map_rotations", "body_fk", "predict",
    }, ns)
    retained = json.loads((worker / "worker/cmu_lightweight.json").read_bytes())
    ns["calibration"] = lambda: (np.array(retained["rotations"]), retained)
    np.testing.assert_array_equal(calibration()[0], ns["calibration"]()[0])
    return ns, reader, utils, asf


def reference_source(ns, reader, utils, asf, bvh, asf_path, sid, fps):
    ns["validate_bvh_header"](bvh)
    anim = reader.read_bvh(str(bvh))
    assert ns["_looks_like_cmu_converter_tpose"](anim)
    ns["_hierarchy"](anim.bones, anim.parents)
    geometry = asf.restore_geometry(anim, dict(assetFile=str(bvh), cmuSequenceId=sid,
                                               cmuAsfFile=str(asf_path)))
    q, positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    canonical = ns["WORLD"] @ Rotation.from_quat(q[0], scalar_first=True).as_matrix()
    world = Rotation.from_matrix(ns["WORLD"])
    q = (world * Rotation.from_quat(q[1:].reshape(-1, 4), scalar_first=True))
    q = q.as_quat(scalar_first=True).reshape(-1, 31, 4)
    positions = world.apply(positions[1:].reshape(-1, 3)).reshape(-1, 31, 3) * geometry["metersPerUnit"]
    rest = world.apply(ns["hierarchy_rest_positions"](anim.offsets, anim.parents)) * geometry["metersPerUnit"]
    ids = np.arange(len(q))
    return dict(source_names=tuple(anim.bones), source_parents=anim.parents, source_rest=rest,
                source_quaternions=q, source_positions=positions, source_frame_ids=ids,
                times=ids / fps), canonical, geometry


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def run(workspace, report_path, export_dir=None):
    worker = workspace / "motion-hangar-worker"
    ns, reader, utils, asf_module = reference(worker)
    timing = workspace / (
        "artifacts/cmu-lightweight-consolidation-20261016/"
        "pair-preparation-v1/design-02/metadata/timing.json")
    model = workspace / "models/smplx/SMPLX_NEUTRAL.npz"
    target, _ = model_rest(model)
    # Independently regress the frozen reference rest; no target rest arrays are saved.
    with np.load(model, allow_pickle=False) as data:
        reference_rest = data["J_regressor"][:22].astype(float) @ data["v_template"].astype(float)
    np.testing.assert_array_equal(target, reference_rest)
    records = []
    for sid in IDS:
        candidates = [workspace / "artifacts/cmu-full-fit-20261015/sources" / (sid + ".bvh"),
                      workspace / "artifacts/umr-production-20261015/sources" / (sid + ".bvh")]
        bvh = next(path for path in candidates if path.is_file())
        asf_path = workspace / "datasets/cmu-smplx-v3/asf" / (sid.split("_")[0] + ".asf")
        result = convert(bvh, asf_path, model, timing_manifest=timing, sequence_id=sid)
        pose = np.concatenate((result.root_orient[:, None], result.pose_body.reshape(-1, 21, 3)), axis=1)
        source, canonical, _ = reference_source(ns, reader, utils, asf_module, bvh, asf_path,
                                               sid, result.mocap_frame_rate)
        profile = ns["build_profile"](source["source_names"], source["source_parents"],
                                       source["source_rest"], canonical, reference_rest)
        expected = ns["predict"](source, profile)
        np.testing.assert_allclose(pose, expected["pose"], atol=3e-11, rtol=0)
        np.testing.assert_allclose(result.trans, expected["trans"], atol=3e-12, rtol=0)
        np.testing.assert_array_equal(result.betas, expected["beta"])
        joints, mapped_world = body_fk(pose, target)
        joints += result.trans[:, None]
        np.testing.assert_allclose(joints, expected["joints"], atol=3e-12, rtol=0)
        # Exercise independent BVH, geometry, calibration and profile construction, too.
        anim = read_bvh(bvh)
        restore_geometry(anim, asf_path)
        globals_, _ = forward_kinematics(anim)
        actual_canonical = WORLD @ globals_[0]
        actual_rest = (rest_positions(anim.offsets, anim.parents) @ WORLD.T
                       * result.provenance["geometry"]["metersPerUnit"])
        actual_offsets = build_offsets(anim.bones, anim.parents, actual_rest, actual_canonical, target)
        np.testing.assert_allclose(actual_canonical, canonical, atol=3e-14, rtol=0)
        np.testing.assert_allclose(actual_rest, source["source_rest"], atol=3e-14, rtol=0)
        np.testing.assert_allclose(actual_offsets, profile["geometricOffsets"], atol=3e-14, rtol=0)
        _, reference_world = ns["body_fk"](expected["pose"], reference_rest)
        row = dict(sequenceId=sid, nativeFrames=len(pose), nativeFps=result.mocap_frame_rate,
                   allNativeFramesCompared=True, reference="frozen Worker pure predict; raw original BVH",
                   maxPoseAbsRad=max_abs(pose, expected["pose"]),
                   maxTranslationAbsM=max_abs(result.trans, expected["trans"]),
                   maxJointAbsM=max_abs(joints, expected["joints"]),
                   maxWorldMatrixAbs=max_abs(mapped_world, reference_world),
                   maxCanonicalMatrixAbs=max_abs(actual_canonical, canonical),
                   maxGeometricOffsetAbs=max_abs(actual_offsets, profile["geometricOffsets"]),
                   inputSha256={key: value["sha256"] for key, value in result.provenance["inputs"].items()})
        # Explicit native FPS gives the same arrays as the trusted manifest.
        explicit = convert(bvh, asf_path, model, fps=result.mocap_frame_rate, sequence_id=sid)
        for key, value in result.arrays().items():
            np.testing.assert_array_equal(explicit.arrays()[key], value)
        row["explicitFpsManifestArrayEquality"] = True
        if export_dir is not None:
            output, _ = save(result, export_dir / (sid + ".npz"))
            row["localOutput"] = str(output)
        records.append(row)
        print(json.dumps(row), flush=True)
    for relative, expected in FROZEN.items():
        assert file_sha256(worker / relative) == expected, relative
    assert "torch" not in sys.modules
    report = dict(schemaVersion="cmu-standalone-external-parity/1",
                  status="passed", nativeSampling=True, comparisonCount=4,
                  workerFileSha256=FROZEN, matricesCopiedFromLicensedModel=False,
                  learnedCalibrationOnly=True, anatomicalAccuracyMeasured=False,
                  productionPoliciesReplayed=False, cases=records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="External fixture workspace root")
    parser.add_argument("--report", type=Path, default=REPO / "test-output/external-parity.json")
    parser.add_argument("--export-dir", type=Path, help="Optional ignored local NPZs for downstream import checks")
    args = parser.parse_args()
    run(args.workspace.resolve(), args.report.resolve(),
        args.export_dir.resolve() if args.export_dir else None)


if __name__ == "__main__":
    main()
