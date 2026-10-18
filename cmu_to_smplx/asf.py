"""Official ASF unit conversion and cgspeed omitted branch translations."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .conventions import require

OFFSETS = {
    "LeftUpLeg": "lhipjoint", "RightUpLeg": "rhipjoint",
    "LeftLeg": "lfemur", "RightLeg": "rfemur",
    "LeftFoot": "ltibia", "RightFoot": "rtibia",
    "LeftToeBase": "lfoot", "RightToeBase": "rfoot",
    "Spine": "lowerback", "Spine1": "upperback",
    "Neck1": "lowerneck", "Head": "upperneck",
    "LeftArm": "lclavicle", "RightArm": "rclavicle",
    "LeftForeArm": "lhumerus", "RightForeArm": "rhumerus",
    "LeftHand": "lradius", "RightHand": "rradius",
}
BRANCH_GROUPS = {
    "thorax": ("Neck", "LeftShoulder", "RightShoulder"),
    "lwrist": ("LeftFingerBase", "LThumb"),
    "rwrist": ("RightFingerBase", "RThumb"),
}
BRANCH_PARENTS = {"thorax": "Spine1", "lwrist": "LeftHand", "rwrist": "RightHand"}


def read_asf(path):
    section, bone = None, None
    units, bones, parents = {}, {}, {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if fields[0].startswith(":"):
            section = fields[0]
        elif section == ":units":
            require(len(fields) >= 2, "Incomplete ASF units")
            units[fields[0]] = fields[1]
        elif section == ":bonedata":
            if fields[0] == "begin":
                require(bone is None, "Nested ASF bone")
                bone = {}
            elif fields[0] == "end":
                require(bone is not None and {"name", "direction", "length"} <= bone.keys(),
                        "Incomplete ASF bone")
                require(bone["name"] not in bones, "Duplicate ASF bone")
                bones[bone["name"]] = bone
                bone = None
            elif bone is not None:
                if fields[0] == "name":
                    require(len(fields) == 2, "Invalid ASF bone name")
                    bone["name"] = fields[1]
                elif fields[0] == "direction":
                    bone["direction"] = [float(x) for x in fields[1:]]
                elif fields[0] == "length":
                    require(len(fields) == 2, "Invalid ASF bone length")
                    bone["length"] = float(fields[1])
        elif section == ":hierarchy" and fields[0] not in ("begin", "end"):
            for child in fields[1:]:
                require(child not in parents, "Duplicate ASF parent")
                parents[child] = fields[0]
    require(bone is None, "Unclosed ASF bone")
    length_unit = float(units.get("length", 0))
    require(units.get("angle") == "deg" and np.isfinite(length_unit) and length_unit > 0,
            "Unsupported ASF units")
    require(set(OFFSETS.values()) | set(BRANCH_GROUPS) <= bones.keys(),
            "Incomplete CMU ASF body geometry")
    for name, value in bones.items():
        direction = np.asarray(value["direction"])
        require(direction.shape == (3,) and np.isfinite(direction).all()
                and np.isfinite(value["length"]) and value["length"] > 0
                and abs(np.linalg.norm(direction) - 1) <= 2e-4,
                "Invalid ASF bone geometry: " + name)
        seen, parent = {name}, parents.get(name)
        while parent != "root":
            require(parent is not None and parent not in seen and parent in bones,
                    "Invalid ASF hierarchy")
            seen.add(parent)
            parent = parents.get(parent)
    return dict(bones=bones, parents=parents, metersPerUnit=.0254 / length_unit)


def restore_geometry(anim, asf_path):
    document = read_asf(asf_path)
    bones = document["bones"]
    index = {name: j for j, name in enumerate(anim.bones)}
    residuals = {}
    for target, source in OFFSETS.items():
        vector = np.asarray(bones[source]["direction"]) * bones[source]["length"]
        residuals[target] = float(np.linalg.norm(anim.offsets[index[target]] - vector))
    require(max(residuals.values()) <= 2e-4,
            "BVH skeleton differs from supplied ASF: " + max(residuals, key=residuals.get))
    restored, updates = {}, []
    for bone, branches in BRANCH_GROUPS.items():
        vector = np.asarray(bones[bone]["direction"]) * bones[bone]["length"]
        states = []
        for name in branches:
            j = index[name]
            require(int(anim.parents[j]) == index[BRANCH_PARENTS[bone]],
                    "Unexpected cgspeed branch parent: " + name)
            offset = anim.offsets[j]
            states.append("missing" if np.linalg.norm(offset) < 2e-4 else
                          "present" if np.linalg.norm(offset - vector) < 2e-4 else "invalid")
        require(len(set(states)) == 1 and states[0] != "invalid",
                "Ambiguous branch offsets: " + bone)
        restored[bone] = dict(restored=states[0] == "missing",
                              lengthM=float(np.linalg.norm(vector) * document["metersPerUnit"]))
        if states[0] == "missing":
            updates.extend((index[name], vector) for name in branches)
    for j, vector in updates:
        anim.offsets[j] += vector
        anim.pos[:, j] += vector
    # This source-only digest deliberately omits vectors from the output metadata.
    geometry = dict(bones={name: bones[name] for name in sorted(bones)},
                    parents=document["parents"], metersPerUnit=document["metersPerUnit"])
    return dict(method="cmu-official-asf-geometry/2",
                metersPerUnit=document["metersPerUnit"], restoredBranches=restored,
                maxExistingOffsetDifferenceM=max(residuals.values()) * document["metersPerUnit"],
                sourceGeometrySha256=hashlib.sha256(
                    json.dumps(geometry, sort_keys=True).encode()).hexdigest(),
                identityVerifiedBy="all-18-body-offset-vectors")
