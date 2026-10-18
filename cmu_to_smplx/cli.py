"""Command-line entry point."""

import argparse
import json
import sys

from .converter import convert, save


def main(argv=None):
    parser = argparse.ArgumentParser(description="Map complete cgspeed CMU BVH / ASF to native SMPL-X body22.")
    parser.add_argument("bvh", help="Original complete31-joint BVH with its prepended calibration row")
    parser.add_argument("--asf", required=True, help="Matching official CMU subject ASF")
    parser.add_argument("--model", required=True, help="Licensed SMPLX_NEUTRAL.npz")
    parser.add_argument("-o", "--output", required=True, help="Output .npz (with .json provenance)")
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument("--fps", type=float, help="Explicit original native action FPS")
    timing.add_argument("--timing-manifest", help="Operator-trusted cmu-source-timing/1 JSON catalog")
    parser.add_argument("--sequence-id", help="CMU identity, e.g.02_01; resolves timing aliases")
    args = parser.parse_args(argv)
    try:
        result = convert(args.bvh, args.asf, args.model, fps=args.fps,
                         timing_manifest=args.timing_manifest, sequence_id=args.sequence_id)
        output, sidecar = save(result, args.output)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print("cmu-to-smplx: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(dict(output=str(output), provenance=str(sidecar),
                          frames=len(result.trans), fps=result.mocap_frame_rate)))
    return 0
