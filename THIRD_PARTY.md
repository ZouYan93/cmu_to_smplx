# Provenance And Third-Party Materials

## Standalone Implementation

The source implements the retained `cmu-fixed-route3/1` formulas, source
semantics, calibration-row signature and ASF restoration conventions reviewed
in `worker/cmu_lightweight.py`, `cmu_asf.py`, `cmu_fixed_basis.py`,
`source_adapter.py` and `cmu_timing.py` on 2026-10-18. The BVH parser is an
independent implementation using NumPy and SciPy. No vendor reader or Worker
module is bundled or needed at runtime.

The optional external parity runner reads the operator's hash-frozen local
Worker files and original vendor reader as test references. Those source files
are not included here. Its source hashes are recorded in the runner and the
result report. The reference vendor GMR checkout carries a notice identifying
Copyright 2025 Yanjieze and an MIT permission notice. This statement concerns
the local test reference; it does not relicense the vendor's data assets.

NumPy and SciPy are separately installed dependencies with their own licenses
and notices. They are not vendored into this repository.

## Learned Calibration

The only distributed numerical parameter asset is
`cmu_to_smplx/calibration.json`: 22 shared 3x3 proper rotations (66 rotational
degrees of freedom), zero target-shape coefficients and provenance metadata.
The matrices were extracted verbatim from the retained Worker configuration:

| Source artifact | SHA256 |
| --- | --- |
| Worker calibration configuration | `6a20ea13a43b26400988eccfef4fa4bacebdd985713ffd863722526ec3986152` |
| Retained calibration-01 model | `e07a026e0e8be2ca88dfc2c6db7495a08fda6ae0093699dbb58f1a961c0a5c2c` |
| Retained geometry code | `4e32fb52f2da7cb94bda05f71bb3d626b3df60b6461bae8c860d9526c7933faf` |
| Retained calibration protocol | `e85fff5284c46fc34680a70b5a4434bfd51bf635a821b5d41de5b31d2e71b198` |

Offline calibration used 1,242 actions from 60 capture groups and 39,600 selected
frames against AMASS fitted parameters. Its objective combined root-relative
joint squared error, local rotation chordal squared error and increment
regularization. Groups, actions and selected frames were weighted equally
within their respective levels.

Prior held evidence had been inspected. The calibration was frozen before
subsequent held-target evaluation, which was not a blind final test. These
learned parameters are neither anatomical ground truth nor a claim of
independent person separation.

There are no copied licensed model-rest arrays, vertices, shape directions,
joint regressors, subject ASF vectors, or motion samples in the configuration.
Learned rotations depend statistically on upstream fitted data. Their
redistribution/use rights are not established by the MIT source-code license;
the repository makes no independent grant of upstream rights. The owner and
users must hold applicable permissions for their intended use.

## Operator-Supplied Data And Model

CMU cgspeed BVH and official ASF are supplied externally. Relevant source
identifiers are `cgspeed.com` (CMU BVH conversion) and `mocap.cs.cmu.edu`
(original data, ASF and unit documentation).

SMPL-X and AMASS have separate access and use terms, obtainable through their
official project sites `smpl-x.is.tue.mpg.de` and `amass.is.tue.mpg.de`. No
assertion that those assets are MIT licensed is made here. This repository
provides no model, capture, ASF or AMASS archive and no credential mechanism.

The runtime computes neutral zero16 rest joints from the user's licensed
`SMPLX_NEUTRAL.npz`. Only input hashes enter JSON provenance; rest-joint
coordinates are not serialized there. Converted user motion files are local
outputs excluded by `.gitignore`.
