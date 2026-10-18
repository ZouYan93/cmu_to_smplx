"""Names and topology only; these constants contain no model geometry."""

import numpy as np
from scipy.spatial.transform import Rotation

METHOD = "cmu-fixed-route3/1"
MODEL_SHA256 = "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992"
SOURCE_CONFIG_SHA256 = "6a20ea13a43b26400988eccfef4fa4bacebdd985713ffd863722526ec3986152"
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
BODY_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
ROTATION_SOURCES = (
    "Hips", "LeftUpLeg", "RightUpLeg", "LowerBack", "LeftLeg", "RightLeg",
    "Spine", "LeftFoot", "RightFoot", "Spine1", "LeftToeBase", "RightToeBase",
    "Neck1", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
    "LeftHand", "RightHand", "LeftFingerBase", "RightFingerBase",
)
POSITION_SOURCES = ROTATION_SOURCES[:18] + (
    "LeftForeArm", "RightForeArm", "LeftFingerBase", "RightFingerBase",
)
SOURCE_PARENTS = {
    "Hips": None, "LHipJoint": "Hips", "RHipJoint": "Hips",
    "LeftUpLeg": "LHipJoint", "RightUpLeg": "RHipJoint",
    "LeftLeg": "LeftUpLeg", "RightLeg": "RightUpLeg",
    "LeftFoot": "LeftLeg", "RightFoot": "RightLeg",
    "LeftToeBase": "LeftFoot", "RightToeBase": "RightFoot",
    "LowerBack": "Hips", "Spine": "LowerBack", "Spine1": "Spine",
    "Neck": "Spine1", "Neck1": "Neck", "Head": "Neck1",
    "LeftShoulder": "Spine1", "RightShoulder": "Spine1",
    "LeftArm": "LeftShoulder", "RightArm": "RightShoulder",
    "LeftForeArm": "LeftArm", "RightForeArm": "RightArm",
    "LeftHand": "LeftForeArm", "RightHand": "RightForeArm",
    "LeftFingerBase": "LeftHand", "RightFingerBase": "RightHand",
    "LeftHandIndex1": "LeftFingerBase", "RightHandIndex1": "RightFingerBase",
    "LThumb": "LeftHand", "RThumb": "RightHand",
}
LIMB_EDGES = ((1, 4), (2, 5), (4, 7), (5, 8), (16, 18), (17, 19), (18, 20), (19, 21))
WORLD = Rotation.from_euler("x", 90, degrees=True).as_matrix()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_hierarchy(names, parents):
    names, parents = tuple(names), np.asarray(parents)
    require(len(names) == len(set(names)) and set(names) == set(SOURCE_PARENTS)
            and names[0] == "Hips" and parents.shape == (31,)
            and parents.dtype.kind in "iu",
            "Expected the complete31-joint cgspeed CMU hierarchy")
    for j, name in enumerate(names):
        parent = SOURCE_PARENTS[name]
        expected = -1 if parent is None else names.index(parent)
        require(int(parents[j]) == expected and expected < j,
                "Unexpected source parent: " + name)
