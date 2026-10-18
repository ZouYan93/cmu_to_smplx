# Local Validation

Run on 2026-10-18 using Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3.

`python -m unittest discover -s tests -v`: **16 tests passed**.
The suite covers synthetic BVH parsing and Euler order, calibration-row
requirements, semantic source maps, mixed/present/missing ASF branches,
native-clock authority and hash/count/evidence errors, model-regressor
semantics, full-calibration-file tampering, ordered rotation products,
framewise independence, preserved native step magnitudes, safe export and
imports without Worker, GMR or Torch.

`python tests/external_parity.py --workspace <external workspace>`:
**four raw-original full-native parity checks passed**.

| Action | Native Frames | FPS | Max Pose Difference (rad) | Max Translation Difference (m) | Max FK Difference (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `02_01` | 343 | 120 | 1.33e-15 | 6.66e-16 | 1.33e-15 |
| `02_05` | 1854 | 120 | 2.22e-15 | 3.89e-16 | 1.36e-15 |
| `06_14` | 479 | 120 | 1.78e-15 | 4.44e-16 | 1.33e-15 |
| `35_01` | 358 | 120 | 1.44e-15 | 8.88e-16 | 1.44e-15 |

All 3,034 native action frames were compared directly with frozen Worker pure
prediction. The external harness uses the original reader, official ASF
restoration, calibration-row orientation and static-profile formulas as
independent preprocessing references. It executes the hash-frozen
`cmu_lightweight.predict` functions without production policy routing.
It does not compare only 30 Hz exports.

The packaged learned matrices equal the frozen Worker configuration
element-for-element. Every explicit-FPS output array equals its corresponding
trusted-manifest output array. Source rest geometry, canonical rotations,
geometric offsets and body-global matrices also pass numeric parity.

The ignored local `test-output/external-parity.json` report records input and
reference-source hashes and scalar errors. The optional ignored native exports
include `up_axis=Z` and `gender=neutral`, with no object arrays.
A real `python -m cmu_to_smplx` export of `02_01` also completed at 120 FPS
with 343 native frames, using an explicit FPS argument.

The wheel was built and installed into a separate directory. Its installed CLI
converted the same original action from outside this repository. The existing
Worker `processing.inspect_file` accepted the generated NPZ as343frames/120Hz;
`motion_filter.native_surface` imported a0.5-second window as15frames/30Hz,
preserving Z-up, neutral gender and16shape coefficients. This checks the file
interface only; the source converter itself performs no resampling.

This validates numerical compatibility for the four fixtures. Anatomical
accuracy, the full 109-group family, Worker quality policies, robot behavior and
production deployment are outside this test result. Licensed geometry and
capture samples are absent from the distributed package.
