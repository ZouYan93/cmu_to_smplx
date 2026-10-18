from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from cmu_to_smplx import convert, save
from cmu_to_smplx import mapping
from cmu_to_smplx.asf import BRANCH_GROUPS, restore_geometry
from cmu_to_smplx.bvh import forward_kinematics, has_calibration_row, read_bvh
from cmu_to_smplx.conventions import (
    PARENTS, POSITION_SOURCES, ROTATION_SOURCES, WORLD, validate_hierarchy,
)
from cmu_to_smplx.timing import resolve_timing
from tests.synthetic import toy_rest, write_inputs

ROOT = Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def setUp(self):
        directory = ROOT / "test-output"
        directory.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bvh, self.asf, self.rows = write_inputs(self.root)

    def conversion(self, **kwargs):
        model = self.root / "model.npz"
        model.write_bytes(b"toy model placeholder")
        with patch("cmu_to_smplx.converter.model_rest",
                   return_value=(toy_rest(), mapping.file_sha256(model))):
            return convert(self.bvh, self.asf, model, **kwargs)

    def test_exact_semantic_sources(self):
        self.assertEqual(ROTATION_SOURCES[18:],
                         ("LeftHand", "RightHand", "LeftFingerBase", "RightFingerBase"))
        self.assertEqual(POSITION_SOURCES[18:],
                         ("LeftForeArm", "RightForeArm", "LeftFingerBase", "RightFingerBase"))

    def test_calibration_has_only_learned_rotations(self):
        rotations, data, digest = mapping.calibration()
        self.assertEqual(rotations.shape, (22, 3, 3))
        self.assertEqual(digest, mapping.CALIBRATION_SHA256)
        self.assertEqual(data["targetBetas"], [0.] * 16)
        self.assertTrue(data["provenance"]["learnedOffline"])
        self.assertFalse(data["provenance"]["geometryIncluded"])
        np.testing.assert_allclose(np.linalg.det(rotations), 1., atol=1e-12)
        self.assertEqual(set(data), {
            "schemaVersion", "method", "sourceConfigSha256", "retainedModelSha256",
            "retainedGeometrySha256", "retainedProtocolSha256", "neutralModelSha256",
            "targetBetas", "provenance", "rotations",
        })

    def test_proper_rotation_tampering_rejected(self):
        _, data, _ = mapping.calibration()
        data["rotations"][0] = Rotation.from_rotvec([.1, .2, .3]).as_matrix().tolist()
        resource = self.root / "calibration.json"
        resource.write_text(json.dumps(data), encoding="ascii")
        with patch.object(mapping, "files", return_value=self.root):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                mapping.calibration()

    def test_bvh_euler_order_and_fk(self):
        anim = read_bvh(self.bvh)
        self.assertTrue(has_calibration_row(anim))
        self.assertEqual(anim.quats.shape, (9, 31, 4))
        j = anim.bones.index("LeftHand")
        z, y, x = self.rows[3, 3 + j * 3:6 + j * 3]
        expected = (Rotation.from_euler("z", z, degrees=True)
                    * Rotation.from_euler("y", y, degrees=True)
                    * Rotation.from_euler("x", x, degrees=True))
        got = Rotation.from_quat(anim.quats[3, j], scalar_first=True)
        np.testing.assert_allclose(got.as_matrix(), expected.as_matrix(), atol=1e-15)
        world, points = forward_kinematics(anim)
        parent = anim.parents[j]
        np.testing.assert_allclose(world[3, j], world[3, parent] @ got.as_matrix(), atol=1e-15)
        np.testing.assert_allclose(points[3, j],
                                   points[3, parent] + world[3, parent] @ anim.offsets[j], atol=1e-15)

    def test_hierarchy_and_channels_refused(self):
        text = self.bvh.read_text()
        for old, new in (("JOINT LeftHand", "JOINT Unknown"),
                         ("Zrotation Yrotation Xrotation", "Xrotation Yrotation Zrotation"),
                         ("Frames: 9", "Frames: 10")):
            with self.subTest(old=old):
                bad = self.root / "bad.bvh"
                bad.write_text(text.replace(old, new, 1), encoding="ascii")
                with self.assertRaises(ValueError):
                    read_bvh(bad)
        anim = read_bvh(self.bvh)
        parents = anim.parents.copy()
        parents[anim.bones.index("LeftFingerBase")] = 0
        with self.assertRaises(ValueError):
            validate_hierarchy(anim.bones, parents)
        with self.assertRaises(ValueError):
            validate_hierarchy(anim.bones[:-1], anim.parents[:-1])

    def test_calibration_removal_is_explicit(self):
        text = self.bvh.read_text().splitlines()
        motion = text.index("MOTION")
        text[motion + 1] = "Frames: 8"
        del text[motion + 3]
        self.bvh.write_text("\n".join(text) + "\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "calibration row"):
            self.conversion(fps=120.)

    def test_asf_geometry_and_idempotence(self):
        anim = read_bvh(self.bvh)
        info = restore_geometry(anim, self.asf)
        self.assertEqual(set(info["restoredBranches"]), set(BRANCH_GROUPS))
        self.assertTrue(all(row["restored"] for row in info["restoredBranches"].values()))
        before = anim.pos.copy()
        info = restore_geometry(anim, self.asf)
        self.assertTrue(all(not row["restored"] for row in info["restoredBranches"].values()))
        np.testing.assert_array_equal(anim.pos, before)
        self.assertAlmostEqual(info["metersPerUnit"], .0254 / .45)

    def test_asf_mixed_branches_and_mismatched_subject_refused(self):
        anim = read_bvh(self.bvh)
        anim.offsets[anim.bones.index("LeftShoulder")] = [1., 2., 3.]
        before = anim.pos.copy()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            restore_geometry(anim, self.asf)
        np.testing.assert_array_equal(anim.pos, before)
        anim = read_bvh(self.bvh)
        anim.offsets[anim.bones.index("LeftUpLeg")] += [.001, 0, 0]
        with self.assertRaisesRegex(ValueError, "differs"):
            restore_geometry(anim, self.asf)

    def test_existing_branch_vectors_accepted(self):
        bvh, asf, _ = write_inputs(self.root / "present", branches_present=True)
        anim = read_bvh(bvh)
        info = restore_geometry(anim, asf)
        self.assertTrue(all(not row["restored"] for row in info["restoredBranches"].values()))

    def test_full_native_root_units_and_export(self):
        result = self.conversion(fps=60.)
        self.assertEqual(result.root_orient.shape, (8, 3))
        self.assertEqual(result.pose_body.shape, (8, 63))
        expected_hips = self.rows[1:, :3] @ WORLD.T * (.0254 / .45)
        np.testing.assert_allclose(result.trans + toy_rest()[0], expected_hips, atol=1e-15)
        self.assertEqual(result.mocap_frame_rate, 60.)
        self.assertEqual(result.provenance["nativeFrameIdRange"], [0, 7])
        self.assertFalse(result.provenance["resampled"])
        output, sidecar = save(result, self.root / "out.npz")
        with np.load(output, allow_pickle=False) as data:
            self.assertEqual(set(data.files), {
                "root_orient", "pose_body", "trans", "betas", "mocap_frame_rate", "up_axis", "gender",
            })
            self.assertEqual(str(data["up_axis"]), "Z")
            self.assertEqual(str(data["gender"]), "neutral")
            self.assertTrue(all(data[key].dtype.kind != "O" for key in data.files))
            for key in ("root_orient", "pose_body", "trans", "betas", "mocap_frame_rate"):
                self.assertTrue(np.isfinite(data[key]).all())
            np.testing.assert_array_equal(data["betas"], np.zeros(16))
        self.assertEqual(json.loads(sidecar.read_text())["nativeFrames"], 8)
        with self.assertRaisesRegex(ValueError, "already exists"):
            save(result, output)
        # A failed JSON serialization must leave neither output file.
        with self.assertRaises(ValueError):
            save(replace(result, provenance={"bad": float("nan")}), self.root / "nan.npz")
        self.assertFalse((self.root / "nan.npz").exists())

    def test_framewise_mapping_products_and_crop_invariance(self):
        rng = np.random.default_rng(609)
        source = Rotation.random(22 * 7, random_state=rng).as_matrix().reshape(7, 22, 3, 3)
        offsets = Rotation.random(22, random_state=rng).as_matrix()
        corrections = mapping.calibration()[0]
        pose = mapping.map_rotations(source, offsets, corrections)
        _, world = mapping.body_fk(pose, toy_rest())
        np.testing.assert_allclose(world, source @ offsets @ corrections, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(mapping.map_rotations(source[2:5], offsets, corrections), pose[2:5])
        # Native local foot step magnitudes survive the constant calibration products.
        raw = mapping.map_rotations(source, offsets, np.broadcast_to(np.eye(3), (22, 3, 3)))
        for j in (7, 8, 10, 11):
            a = Rotation.from_rotvec(raw[:, j])
            b = Rotation.from_rotvec(pose[:, j])
            np.testing.assert_allclose((a[:-1].inv() * a[1:]).magnitude(),
                                       (b[:-1].inv() * b[1:]).magnitude(), atol=3e-15)

    def test_invalid_rotations_and_swing(self):
        for matrix in (np.diag([-1., 1., 1.]), np.full((3, 3), np.nan), np.zeros((3, 3))):
            with self.assertRaises(ValueError):
                mapping.proper(matrix)
        for a, b in ((np.zeros(3), np.ones(3)), (np.array([1., 0, 0]), np.array([-1., 0, 0]))):
            with self.assertRaises(ValueError):
                mapping.swing(a, b)

    def test_native_clock_requires_authority(self):
        for kwargs in ({}, {"fps": 120., "manifest": "unused"}, {"fps": True},
                       {"fps": float("nan")}, {"fps": 0.}, {"fps": 241.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resolve_timing(self.bvh, 9, .0333333, **kwargs)
        for rate in (30., 60., 120.):
            info = resolve_timing(self.bvh, 9, .0333333, fps=rate)
            self.assertEqual(info["nativeFps"], rate)
            self.assertEqual(info["authority"], "caller assertion")
            self.assertAlmostEqual(info["headerFps"], 1 / .0333333)

    def test_timing_hash_count_evidence_and_aliases(self):
        raw = self.bvh.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        row = dict(sequenceId="02_01", sourceGitBlobSha1=blob, sourceFps=120.,
                   sourceFrames=8, prependedFrames=1, evidenceKind="original-c3d-metadata",
                   evidenceSha256="a" * 64)
        manifest = self.root / "clock.json"

        def write(rows):
            manifest.write_text(json.dumps(dict(schemaVersion="cmu-source-timing/1", entries=rows)))

        write([row])
        clock = resolve_timing(self.bvh, 9, .0333333, manifest=manifest)
        self.assertEqual(clock["nativeFps"], 120.)
        for field, value in (("sourceGitBlobSha1", "0" * 40), ("sourceFps", True),
                             ("sourceFrames", 9), ("evidenceSha256", "invalid"),
                             ("prependedFrames", 0)):
            write([dict(row, **{field: value})])
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_timing(self.bvh, 9, .0333333, manifest=manifest)
        write([row, dict(row, sequenceId="02_02", sourceFps=60.)])
        with self.assertRaisesRegex(ValueError, "different clocks"):
            resolve_timing(self.bvh, 9, .0333333, manifest=manifest)
        clock = resolve_timing(self.bvh, 9, .0333333, manifest=manifest, sequence_id="02_02")
        self.assertEqual(clock["nativeFps"], 60.)

    def test_model_regression_and_model_hash(self):
        file = self.root / "toy-model.npz"
        # This is generated toy geometry, with no values from SMPL-X.
        kintree = np.vstack((np.asarray(PARENTS, dtype=int), np.arange(22)))
        np.savez(file, kintree_table=kintree, J_regressor=np.eye(22), v_template=toy_rest())
        with self.assertRaisesRegex(ValueError, "SHA256"):
            mapping.model_rest(file)
        with patch.object(mapping, "MODEL_SHA256", mapping.file_sha256(file)):
            rest, _ = mapping.model_rest(file)
        np.testing.assert_array_equal(rest, toy_rest())

    def test_cli_clock_required_and_runtime_independent(self):
        result = subprocess.run([sys.executable, "-m", "cmu_to_smplx", str(self.bvh),
                                 "--asf", str(self.asf), "--model", "unused", "-o", "unused.npz"],
                                capture_output=True, text=True, cwd=ROOT, timeout=20)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--fps", result.stderr)
        probe = subprocess.run([sys.executable, "-c",
                                "import sys,cmu_to_smplx; "
                                "assert not any(n in sys.modules for n in "
                                "('torch','worker','processing','general_motion_retargeting','smplx'))"],
                               capture_output=True, text=True, cwd=ROOT, timeout=20)
        self.assertEqual(probe.returncode, 0, probe.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
