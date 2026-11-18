"""Measure the displayed joint interpolation between saved replay samples.

This checks three interior points per segment by default. It is a sampled
kinematic check, not continuous collision verification or a dynamics test.
The finite ferrule cylinder uses the same dimensions as the Viser viewer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from panda_replay_kinematics import PandaReplayKinematics, _rotation_vector


def check_trajectory(path: str | Path, fractions=(0.25, 0.5, 0.75),
                     max_tip_deviation_m=0.0001, max_orientation_rad=0.01) -> dict:
    """Check interior joint interpolation against the straight tip segment.

    The 0.1 mm deviation and 0.01 rad orientation bounds are replay acceptance
    settings, not claims of physical Panda accuracy. Ink is deposited only at
    the exported contact samples, so intermediate points add no ink events.
    """
    path = Path(path)
    fractions = np.asarray(fractions, dtype=float)
    if (fractions.ndim != 1 or not len(fractions) or not np.all(np.isfinite(fractions))
            or np.any(fractions <= 0) or np.any(fractions >= 1)):
        raise ValueError("fractions must be a nonempty sequence strictly between zero and one")
    if (not np.isfinite(max_tip_deviation_m) or max_tip_deviation_m <= 0
            or not np.isfinite(max_orientation_rad) or max_orientation_rad <= 0):
        raise ValueError("Acceptance limits must be positive finite values")
    with np.load(path, allow_pickle=False) as data:
        q = np.array(data["q"], dtype=float)
        target = np.array(data["target_tip"], dtype=float)
        times = np.array(data["time_s"], dtype=float)
        contact = np.array(data["contact_index"], dtype=int)
        metadata = json.loads(str(np.asarray(data["metadata"]).item())) if "metadata" in data else {}
    count = len(q)
    if (count < 2 or q.shape != (count, 7) or target.shape != (count, 3)
            or times.shape != (count,) or contact.shape != (count,)):
        raise ValueError("Replay requires matching q[N,7], target_tip[N,3], time_s[N], contact_index[N]")
    if not all(np.all(np.isfinite(array)) for array in (q, target, times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory values must be finite and time_s strictly increasing")
    kinematics = PandaReplayKinematics()
    maximum_deviation, maximum_orientation = 0.0, 0.0
    minimum_clearance, minimum_margin = float("inf"), float("inf")
    maximum_deviation_segment, maximum_orientation_segment = 0, 0
    approach_contacts = 0
    for index in range(count - 1):
        for fraction in fractions:
            interpolated_q = (1.0-fraction) * q[index] + fraction * q[index+1]
            hand, rotation, tip = kinematics.forward(interpolated_q)
            cartesian_segment = (1.0-fraction) * target[index] + fraction * target[index+1]
            deviation = float(np.linalg.norm(tip-cartesian_segment))
            orientation = float(np.linalg.norm(_rotation_vector(
                kinematics.TOOL_DOWN_ROTATION @ rotation.T)))
            if deviation > maximum_deviation:
                maximum_deviation, maximum_deviation_segment = deviation, index
            if orientation > maximum_orientation:
                maximum_orientation, maximum_orientation_segment = orientation, index
            # Vertical support of an oriented cylinder: h/2 |cos(a)| + r |sin(a)|.
            ferrule_center = hand + 0.1675 * rotation[:, 2]
            vertical_support = (0.0125 * abs(rotation[2, 2])
                                + 0.0072 * np.sqrt(max(0.0, 1.0-rotation[2, 2]**2)))
            clearance = float(ferrule_center[2]-vertical_support-0.002)
            minimum_clearance = min(minimum_clearance, clearance)
            minimum_margin = min(minimum_margin, float(np.min(np.minimum(
                interpolated_q-kinematics.joint_limits[:, 0],
                kinematics.joint_limits[:, 1]-interpolated_q))))
            # Geometric contact while approaching is expected. The viewer must
            # deform the tuft here even though this is not a recorded ink event.
            approach_contacts += int(contact[index] < 0 and tip[2] < 0.002)
    velocity_ratio = float(np.max(np.abs(np.diff(q, axis=0))
                                 / np.diff(times)[:, None] / kinematics.velocity_limits))
    expected_urdf_hash = metadata.get("urdf_sha256")
    checks = {
        "interior_tip_deviation_within_limit": maximum_deviation <= max_tip_deviation_m,
        "interior_orientation_within_limit": maximum_orientation <= max_orientation_rad,
        "interior_ferrule_above_paper": minimum_clearance > 0.0,
        "interior_joints_inside_limits": minimum_margin >= 0.0,
        "linear_joint_velocity_within_limits": velocity_ratio <= 1.0+1e-9,
        "same_urdf_as_replay": expected_urdf_hash is None or expected_urdf_hash == kinematics.urdf_sha256,
    }
    return {
        "trajectory_file": str(path.resolve()),
        "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "urdf_sha256": kinematics.urdf_sha256,
        "scope": "sampled kinematic interpolation; no continuous collision proof or dynamics validation",
        "ink_semantics": "ink only at exported contact samples; intermediate animation adds no stamps",
        "fractions_per_segment": fractions.tolist(),
        "segments": count-1,
        "interior_samples": (count-1)*len(fractions),
        "max_tip_deviation_from_linear_cartesian_segment_m": maximum_deviation,
        "max_tip_deviation_segment": maximum_deviation_segment,
        "max_orientation_error_rad": maximum_orientation,
        "max_orientation_error_segment": maximum_orientation_segment,
        "minimum_ferrule_clearance_m": minimum_clearance,
        "minimum_joint_limit_margin_rad": minimum_margin,
        "max_joint_velocity_ratio": velocity_ratio,
        "approach_samples_in_geometric_contact": approach_contacts,
        "acceptance_limits": {"tip_deviation_m": max_tip_deviation_m,
                              "orientation_error_rad": max_orientation_rad},
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path, nargs="+", help="saved panda_trajectory.npz files")
    parser.add_argument("--output", type=Path, required=True, help="JSON measurement record")
    parser.add_argument("--max-tip-deviation", type=float, default=0.0001, help="acceptance limit in metres")
    parser.add_argument("--max-orientation", type=float, default=0.01, help="acceptance limit in radians")
    arguments = parser.parse_args()
    results = [check_trajectory(path, max_tip_deviation_m=arguments.max_tip_deviation,
                                max_orientation_rad=arguments.max_orientation)
               for path in arguments.trajectories]
    report = {"status": "passed" if all(result["status"] == "passed" for result in results) else "failed",
              "results": results}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "trajectories": len(results),
                      "output": str(arguments.output.resolve())}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
