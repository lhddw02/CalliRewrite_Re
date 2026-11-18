"""URDF-consistent Panda pose kinematics for offline CalliRewrite replay.

The brush tip is ``panda_hand + tool_length * hand_z``.  All positions are
metres in ``panda_link0`` and all joint angles are radians.  This module reads
joint transforms from an already cached URDF; it never downloads robot assets.
It performs kinematic checks, not dynamics or mesh collision checking.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    return (_rotation(np.array([0.0, 0.0, 1.0]), rpy[2])
            @ _rotation(np.array([0.0, 1.0, 0.0]), rpy[1])
            @ _rotation(np.array([1.0, 0.0, 0.0]), rpy[0]))


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Principal SO(3) logarithm, including the near-pi case."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    if angle < 1e-7:
        return 0.5 * skew
    if np.pi - angle < 1e-5:
        symmetric = (rotation + rotation.T + 2.0 * np.eye(3)) / 4.0
        _, vectors = np.linalg.eigh(symmetric)
        axis = vectors[:, -1]
        if np.dot(axis, skew) < 0.0:
            axis = -axis
        return angle * axis
    return angle / (2.0 * np.sin(angle)) * skew


@dataclass(frozen=True)
class PoseSolution:
    q: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    converged: bool
    iterations: int

    def __post_init__(self):
        self.q.setflags(write=False)


@dataclass(frozen=True)
class PathSolution:
    q: np.ndarray
    achieved_tip: np.ndarray
    position_error_m: np.ndarray
    orientation_error_rad: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray

    def __post_init__(self):
        for array in (self.q, self.achieved_tip, self.position_error_m,
                      self.orientation_error_rad, self.converged, self.iterations):
            array.setflags(write=False)

    @property
    def success(self) -> bool:
        return bool(np.all(self.converged))


class PandaReplayKinematics:
    """Seven-joint Panda FK and bounded, warm-started pose inverse kinematics.

    ``solve_pose`` and ``solve_path`` return explicit convergence diagnostics.
    Callers should reject a path with unconverged samples.  The default fixed
    hand orientation has its Z axis pointing vertically down onto the paper.
    """

    NOMINAL_REST_Q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.785])
    TOOL_DOWN_ROTATION = np.diag([1.0, -1.0, -1.0])

    def __init__(self, tool_length: float = 0.225, urdf_path: str | Path | None = None):
        if not np.isfinite(tool_length) or tool_length <= 0.0:
            raise ValueError("tool_length must be a positive finite distance in metres")
        self.tool_length = float(tool_length)
        self.urdf_path = Path(urdf_path) if urdf_path is not None else self.cached_urdf_path()
        data = self.urdf_path.read_bytes()
        self.urdf_sha256 = hashlib.sha256(data).hexdigest()
        robot = ET.fromstring(data)
        by_child = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
        chain = []
        link = "panda_hand"
        while link != "panda_link0":
            if link not in by_child:
                raise ValueError(f"URDF has no chain from panda_link0 to panda_hand ({link})")
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        chain.reverse()
        self._chain = []
        joint_names, limits, velocity_limits = [], [], []
        for joint in chain:
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            joint_type = joint.get("type")
            axis = np.array([0.0, 0.0, 1.0])
            index = None
            if joint_type == "revolute":
                axis_element = joint.find("axis")
                axis = np.fromstring(axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0", sep=" ")
                axis /= np.linalg.norm(axis)
                index = len(joint_names)
                joint_names.append(joint.get("name"))
                limit = joint.find("limit")
                limits.append([float(limit.get("lower")), float(limit.get("upper"))])
                velocity_limits.append(float(limit.get("velocity")))
            elif joint_type != "fixed":
                raise ValueError(f"Unexpected {joint_type} joint in Panda arm chain")
            self._chain.append((xyz, _rpy_rotation(rpy), axis, index))
        self.joint_names = tuple(joint_names)
        if self.joint_names != tuple(f"panda_joint{i}" for i in range(1, 8)):
            raise ValueError(f"Expected Panda's seven arm joints, got {self.joint_names}")
        self.joint_limits = np.asarray(limits)
        self.JOINT_LIMITS = self.joint_limits
        self.velocity_limits = np.asarray(velocity_limits)

    @staticmethod
    def cached_urdf_path() -> Path:
        path = (Path.home() / ".cache/robot_descriptions/example-robot-data/robots/"
                "panda_description/urdf/panda.urdf")
        if not path.is_file():
            raise FileNotFoundError(
                "Panda URDF is not cached. Load panda_description with robot_descriptions "
                "once, or pass urdf_path pointing to the same Panda URDF used by the viewer."
            )
        return path

    @staticmethod
    def _validate_q(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("q must contain seven finite joint angles")
        return q

    def _forward(self, q: np.ndarray, with_jacobian: bool = False):
        rotation, position = np.eye(3), np.zeros(3)
        origins, axes = [], []
        for xyz, origin_rotation, axis, index in self._chain:
            position = position + rotation @ xyz
            rotation = rotation @ origin_rotation
            if index is not None:
                if with_jacobian:
                    origins.append(position.copy())
                    axes.append(rotation @ axis)
                rotation = rotation @ _rotation(axis, q[index])
        tip = position + self.tool_length * rotation[:, 2]
        if not with_jacobian:
            return position, rotation, tip
        axes = np.asarray(axes)
        jacobian = np.concatenate((np.cross(axes, tip - np.asarray(origins)).T, axes.T), axis=0)
        return position, rotation, tip, jacobian

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._forward(self._validate_q(q))

    # Compatibility with the existing brush visual update helper.
    forward_kinematics = forward

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Geometric tip Jacobian: world XYZ above world angular velocity."""
        return self._forward(self._validate_q(q), with_jacobian=True)[3]

    def solve_pose(self, target_tip: np.ndarray, current_q: np.ndarray | None = None,
                   target_rotation: np.ndarray | None = None, max_iters: int = 100,
                   position_tolerance_m: float = 1e-8,
                   orientation_tolerance_rad: float = 1e-7) -> PoseSolution:
        target = np.asarray(target_tip, dtype=float)
        desired_rotation = (self.TOOL_DOWN_ROTATION if target_rotation is None
                            else np.asarray(target_rotation, dtype=float))
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_tip must be a finite XYZ position in metres")
        if (desired_rotation.shape != (3, 3) or not np.all(np.isfinite(desired_rotation))
                or not np.allclose(desired_rotation.T @ desired_rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(desired_rotation), 1.0, atol=1e-6)):
            raise ValueError("target_rotation must be a proper 3 by 3 rotation matrix")
        if max_iters < 1 or position_tolerance_m <= 0 or orientation_tolerance_rad <= 0:
            raise ValueError("Iteration count and convergence tolerances must be positive")
        lower, upper = self.joint_limits.T
        seed = self.NOMINAL_REST_Q if current_q is None else self._validate_q(current_q)
        q = np.clip(seed.copy(), lower + 1e-6, upper - 1e-6)
        # An angular error of one radian has the weight of 0.2 metres.
        weights = np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.2])
        damping = 0.002
        iterations = 0
        for iterations in range(1, max_iters + 1):
            _, rotation, tip, jacobian = self._forward(q, with_jacobian=True)
            error = np.r_[target - tip, _rotation_vector(desired_rotation @ rotation.T)]
            if (np.linalg.norm(error[:3]) <= position_tolerance_m
                    and np.linalg.norm(error[3:]) <= orientation_tolerance_rad):
                break
            weighted_jacobian = weights[:, None] * jacobian
            weighted_error = weights * error
            delta = weighted_jacobian.T @ np.linalg.solve(
                weighted_jacobian @ weighted_jacobian.T + damping**2 * np.eye(6), weighted_error)
            delta *= min(1.0, 0.22 / max(float(np.linalg.norm(delta)), 1e-12))
            previous_cost = float(weighted_error @ weighted_error)
            improved = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = np.clip(q + scale * delta, lower + 1e-6, upper - 1e-6)
                _, candidate_rotation, candidate_tip = self._forward(candidate)
                candidate_error = weights * np.r_[target - candidate_tip,
                    _rotation_vector(desired_rotation @ candidate_rotation.T)]
                if float(candidate_error @ candidate_error) < previous_cost:
                    q = candidate
                    damping = max(0.0005, damping * 0.8)
                    improved = True
                    break
            if not improved:
                damping = min(0.1, damping * 3.0)
                if damping == 0.1 and np.linalg.norm(delta) < 1e-8:
                    break
        _, rotation, tip = self._forward(q)
        position_error = float(np.linalg.norm(target - tip))
        orientation_error = float(np.linalg.norm(_rotation_vector(desired_rotation @ rotation.T)))
        return PoseSolution(q=q, position_error_m=position_error,
                            orientation_error_rad=orientation_error,
                            converged=bool(position_error <= position_tolerance_m
                                           and orientation_error <= orientation_tolerance_rad),
                            iterations=iterations)

    def solve_path(self, target_tips: np.ndarray, initial_q: np.ndarray | None = None,
                   target_rotation: np.ndarray | None = None, **solver_options) -> PathSolution:
        """Warm-start each sample from its predecessor; report every failed pose."""
        targets = np.asarray(target_tips, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 3 or not np.all(np.isfinite(targets)):
            raise ValueError("target_tips must be a finite N by 3 array")
        count = len(targets)
        q_values, achieved = np.empty((count, 7)), np.empty((count, 3))
        position_error, orientation_error = np.empty(count), np.empty(count)
        converged, iterations = np.empty(count, dtype=bool), np.empty(count, dtype=int)
        seed = initial_q
        for index, target in enumerate(targets):
            result = self.solve_pose(target, seed, target_rotation, **solver_options)
            seed = result.q
            q_values[index] = result.q
            achieved[index] = self.forward(result.q)[2]
            position_error[index] = result.position_error_m
            orientation_error[index] = result.orientation_error_rad
            converged[index], iterations[index] = result.converged, result.iterations
        return PathSolution(q_values, achieved, position_error, orientation_error, converged, iterations)
