# cmu_to_smplx

A small NumPy + SciPy CLI and library for the retained `calibration-01` CMU
cgspeed BVH / official ASF to SMPL-X body22 mapping (`cmu-fixed-route3/1`).
Every native action row is mapped independently. There is no fitting, neural
network, smoothing, source repair, cropping or resampling.

## Install

Python 3.10 or newer:

```bash
python -m pip install .
```

For running directly from this directory:

```bash
python -m pip install -r requirements.txt
python -m cmu_to_smplx --help
```

The runtime has no Worker, server, artifact catalog, GMR, Torch, SMPL-X Python
package or indexed-fit dependency. No remote service is used.

## Inputs

Provide all three files yourself:

- An original Y-up cgspeed CMU BVH with the complete 31-joint hierarchy and the
  converter's original prepended calibration row.
- Its matching official CMU subject ASF, including bone directions, lengths,
  hierarchy and units.
- A licensed `SMPLX_NEUTRAL.npz` with SHA256
  `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`.
  This model revision is pinned to reproduce the retained mapping.

The supported family is the shared 31-joint cgspeed hierarchy represented across
109 capture groups and 97 source geometries in the source-family audit. These
counts describe the audited family, not 109 distinct people or a guarantee of
anatomical accuracy on every action. Local external parity covers four actions.
Arbitrary BVH files, alternate skeletons, channel orders and files with the
calibration row removed are outside this converter's contract.

The converter checks all 18 existing ASF body offsets (tolerance `2e-4` in source
units). It restores omitted thorax and wrist branch translations in memory,
accepts already-present consistent branches and rejects mixed branch states.
ASF length units convert to metres using `0.0254 / length_unit`. Input files
remain unchanged.

## Native Timing

The BVH header often describes a converter clock. Supply the original action's
native FPS explicitly:

```bash
cmu-to-smplx captures/02_01.bvh --asf captures/02.asf \
  --model models/SMPLX_NEUTRAL.npz --fps 120 --output output/02_01.npz
```

`--fps` is a caller assertion, recorded as such. It never comes from the
converter header or a filename. The command removes exactly one recognized
calibration row and retains every remaining row at the supplied rate.

Alternatively, supply an operator-trusted `cmu-source-timing/1` JSON manifest:

```bash
cmu-to-smplx captures/02_01.bvh --asf captures/02.asf \
  --model models/SMPLX_NEUTRAL.npz --timing-manifest timing.json \
  --sequence-id 02_01 --output output/02_01.npz
```

Each manifest entry binds exact BVH bytes through Git blob SHA1 and supplies
`sequenceId`, `sourceFps`, `sourceFrames` (excluding calibration),
`prependedFrames: 1`, `evidenceKind` (`original-c3d-metadata` or
`amass-source-sampling`) and `evidenceSha256`. Conflicting alias clocks require
`--sequence-id`. The manifest is a trusted local assertion, not a signature or
an automatic verification of the cited evidence. This repository includes no
capture timing catalog. Exactly one timing option is required.

## Output And Library

The output is an NPZ plus a JSON provenance sidecar. Existing outputs are
preserved. NPZ arrays load with `allow_pickle=False`:

| Name | Shape | Meaning |
| --- | --- | --- |
| `root_orient` | `(F, 3)` | Root axis-angle, radians |
| `pose_body` | `(F, 63)` | 21 local body rotations, axis-angle |
| `trans` | `(F, 3)` | Root translation, metres |
| `betas` | `(16,)` | All zero, neutral target |
| `mocap_frame_rate` | scalar | Explicit or manifest native FPS |
| `up_axis` | scalar string | `Z` |
| `gender` | scalar string | `neutral` |

Coordinates use the retained positive 90-degree rotation about X, from source
Y-up to right-handed Z-up. Translation is the converted original Hips position
minus the **unrotated** zero16 model rest pelvis. No floor shift, recentering or
actor scale fitting is performed. Frame IDs are `0..F-1` after calibration;
sample times are `frame_id / fps`.

```python
from cmu_to_smplx import convert, save

motion = convert(
    "captures/02_01.bvh",
    "captures/02.asf",
    "models/SMPLX_NEUTRAL.npz",
    fps=120.0,
)
save(motion, "output/02_01.npz")
```

The JSON records input and calibration hashes, timing authority, geometry
restoration, semantic source maps, native frame coverage and inference limits.
It contains no model-rest joint arrays.

## Mapping And Limits

For each target joint, source global rotation is right-multiplied by a static
source/target rest correction and the retained learned rotation. Target locals
come from the ordered parent-global inverse product.

The elbow **rotation** uses raw `LeftHand` / `RightHand` globals, including the
ulna helper. Wrist rotation uses `LeftFingerBase` / `RightFingerBase`. The
separate `POSITION_SOURCES` map uses `ForeArm` elbow positions and `FingerBase`
wrist positions. Substituting `ForeArm` / `Hand` rotation sources changes this
mapping.

Neutral zero16 rest joints are derived at runtime as
`J_regressor[:22].astype(float) @ v_template.astype(float)`.
No SMPL-X vertices, joint-rest arrays, shape directions, model files, ASF files
or captures are included.

`calibration.json` contains 22 shared proper rotation matrices learned offline
against AMASS fitted parameters: 1,242 training actions, 60 capture groups and
39,600 selected frames. These are learned parameters, not a purely geometric
conversion. The package pins the full calibration file SHA256. Editing its
matrices requires an explicit new calibration and version.

Hands and face are unestimated; body22 ends at the wrists. Fixed shape does
not reconstruct individual morphology. Native rotation jumps are preserved.
This tool performs structural/numeric checks; it does not reproduce the
Worker's source-quality or publication policies. Outputs are approximate
mapped parameters, with anatomical accuracy unmeasured and production approval
unset. Matching a predictor establishes numeric compatibility.

## Tests

The default suite generates toy geometry and channels locally:

```bash
python -m unittest discover -s tests -v
```

Full-native external parity is opt-in and requires the original local fixture
workspace; no fixture assets are bundled:

```bash
python tests/external_parity.py --workspace /path/to/external/workspace
```

This runner checks raw originals `02_01`, `02_05`, `06_14`, `35_01` against
hash-frozen Worker pure preprocessing/prediction. It compares every native
pose, translation and body22 FK result, plus static calibration geometry.
The reference code is loaded only by the opt-in test. Its report contains
hashes and scalar errors, with no licensed geometry or motion arrays.

See [VALIDATION.md](VALIDATION.md) for the recorded local results and
[THIRD_PARTY.md](THIRD_PARTY.md) for data/model provenance and licensing scope.
The source-code license is [LICENSE](LICENSE).
