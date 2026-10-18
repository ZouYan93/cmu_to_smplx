"""Explicit native FPS or an operator-trusted hash-bound CMU timing catalog."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .conventions import require


def resolve_timing(bvh, stored_frames, frame_time, *, fps=None, manifest=None, sequence_id=None):
    require((fps is None) != (manifest is None),
            "Provide exactly one of native fps or a trusted timing manifest")
    if sequence_id is not None:
        require(re.fullmatch(r"\d{2,3}_\d{2,3}", str(sequence_id)) is not None,
                "Invalid CMU sequence identity")
    result = dict(headerFps=1 / frame_time, storedFrames=stored_frames,
                  nativeFrames=stored_frames - 1, discardedConverterFrames=1,
                  timeline="source-action-seconds", resampled=False)
    if manifest is None:
        require(type(fps) in (int, float) and np.isfinite(fps) and 1 <= fps <= 240,
                "Native fps must be a finite number between1 and240")
        result.update(method="explicit-native-fps/1", nativeFps=float(fps),
                      authority="caller assertion", sequenceId=sequence_id)
        return result
    path = Path(manifest)
    require(path.is_file() and path.stat().st_size <= 8 * 1024**2,
            "Timing manifest missing or exceeds8MiB")
    raw = path.read_bytes()
    doc = json.loads(raw)
    require(isinstance(doc, dict) and doc.get("schemaVersion") == "cmu-source-timing/1"
            and isinstance(doc.get("entries"), list), "Unsupported timing manifest")
    data = Path(bvh).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    rows = [row for row in doc["entries"] if isinstance(row, dict)
            and row.get("sourceGitBlobSha1") == blob
            and (sequence_id is None or row.get("sequenceId") == sequence_id)]
    require(bool(rows), "Timing manifest has no matching BVH hash / sequence")
    clocks = set()
    for row in rows:
        rate, count = row.get("sourceFps"), row.get("sourceFrames")
        require(type(rate) in (int, float) and np.isfinite(rate) and 1 <= rate <= 240
                and type(count) is int and 2 <= count <= 72000
                and row.get("prependedFrames") == 1
                and re.fullmatch(r"\d{2,3}_\d{2,3}", str(row.get("sequenceId"))) is not None
                and row.get("evidenceKind") in ("amass-source-sampling", "original-c3d-metadata")
                and re.fullmatch(r"[0-9a-f]{64}", str(row.get("evidenceSha256"))) is not None,
                "Invalid timing provenance")
        clocks.add((float(rate), count))
    require(len(clocks) == 1, "Identical BVH aliases have different clocks; provide sequence_id")
    rate, count = clocks.pop()
    require(count + 1 == stored_frames, "Timing manifest frame count differs from BVH")
    result.update(method="cmu-source-timing/1", nativeFps=rate,
                  manifestSha256=hashlib.sha256(raw).hexdigest(), sourceGitBlobSha1=blob,
                  sequenceIds=[row["sequenceId"] for row in rows],
                  evidence=[dict(kind=row["evidenceKind"], sha256=row["evidenceSha256"]) for row in rows])
    return result
