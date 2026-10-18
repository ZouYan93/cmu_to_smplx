"""Generated toy geometry and channels, independent of licensed assets."""

from pathlib import Path

import numpy as np

from cmu_to_smplx.asf import BRANCH_GROUPS, OFFSETS
from cmu_to_smplx.conventions import PARENTS, SOURCE_PARENTS


def toy_rest():
    rest = np.zeros((22, 3), dtype=float)
    for j, parent in enumerate(PARENTS[1:], 1):
        rest[j] = rest[parent] + [.08 + j * .007, .04 + j * .002, .11 + j * .003]
    rest += [.02, -.03, .07]
    return rest


def write_inputs(directory, *, frames=8, branches_present=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vectors = {}
    bone_names = sorted(set(OFFSETS.values()) | set(BRANCH_GROUPS))
    asf = [":version 1.10", ":units", "length 0.45", "angle deg", ":bonedata"]
    for j, name in enumerate(bone_names):
        direction = np.array([1 + j * .07, .4 + j * .05, .2 + j * .03])
        direction /= np.linalg.norm(direction)
        length = 1. + j * .11
        vectors[name] = direction * length
        asf.extend(["begin", "name " + name,
                    "direction " + " ".join(format(x, ".17g") for x in direction),
                    "length " + format(length, ".17g"), "end"])
    asf.extend([":hierarchy", "begin", "root " + " ".join(bone_names), "end"])
    asf_path = directory / "synthetic.asf"
    asf_path.write_text("\n".join(asf) + "\n", encoding="ascii")
    children = {name: [] for name in SOURCE_PARENTS}
    for name, parent in SOURCE_PARENTS.items():
        if parent is not None:
            children[parent].append(name)
    offsets = {name: np.zeros(3) for name in SOURCE_PARENTS}
    offsets.update({target: vectors[source] for target, source in OFFSETS.items()})
    if branches_present:
        for source, targets in BRANCH_GROUPS.items():
            offsets.update({target: vectors[source] for target in targets})
    lines, names = ["HIERARCHY"], []

    def joint(name, depth):
        indent = "  " * depth
        names.append(name)
        lines.extend([indent + ("ROOT " if depth == 0 else "JOINT ") + name, indent + "{",
                      indent + "  OFFSET " + " ".join(format(x, ".17g") for x in offsets[name]),
                      indent + ("  CHANNELS 6 Xposition Yposition Zposition" if depth == 0 else "  CHANNELS 3")
                      + " Zrotation Yrotation Xrotation"])
        for child in children[name]:
            joint(child, depth + 1)
        if not children[name]:
            lines.extend([indent + "  End Site", indent + "  {",
                          indent + "    OFFSET 0.1 0.2 0.3", indent + "  }"])
        lines.append(indent + "}")

    joint("Hips", 0)
    rows = np.zeros((frames + 1, 96))
    rng = np.random.default_rng(608)
    rows[1:, 3:] = rng.normal(0., 17., (frames, 93))
    rows[:, :3] = np.column_stack((np.arange(frames + 1) + 2.,
                                   np.arange(frames + 1) * .4 + 3.,
                                   np.arange(frames + 1) * .2 - 1.))
    for name, angle in (("LeftUpLeg", 21.), ("RightUpLeg", -21.),
                        ("LeftArm", -8.), ("RightArm", 8.)):
        rows[0, 3 + names.index(name) * 3] = angle
    lines.extend(["MOTION", f"Frames: {frames + 1}", "Frame Time: 0.0333333"])
    lines.extend(" ".join(format(value, ".17g") for value in row) for row in rows)
    bvh_path = directory / "synthetic.bvh"
    bvh_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return bvh_path, asf_path, rows
