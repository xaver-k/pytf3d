"""
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


from enum import auto, Enum, unique
from pytf3d.typing import (
    ARRAY_LIKE_1D_T,
    ARRAY_LIKE_2D_T,
    HOMOGENEOUS_MATRIX_T,
    HOMOGENEOUS_VECTOR_T,
    QUATERNION_T,
    ROTATION_MATRIX_T,
    UNIT_QUATERNION_T,
    UNIT_VECTOR_T,
    VECTOR_T,
)
from pytf3d.utils import is_rotation_matrix
from typing import overload, Sequence, Set, Tuple, Union

import numpy as np

QUATERNION_TOLERANCE = 1e-7


@unique
class QuaternionOrder(Enum):
    WXYZ = auto()
    XYZW = auto()


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[[3, 0, 1, 2]]


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[[1, 2, 3, 0]]


class Rotation:
    """
    3D rigid body rotation
    """

    def __init__(self, q: Union[QUATERNION_T, ARRAY_LIKE_1D_T], q_order: QuaternionOrder = QuaternionOrder.WXYZ):
        """
        :param q: quaternion values describing the desired rotation;
                  * 4-dimensional (but passing extra empty axes is possible, they will be squeezed out)
                  * will be normalized internally
        :param q_order: ordering of quaternion values in q
        """

        q_: QUATERNION_T = np.asarray(q, np.float64).squeeze()
        self._raise_if_not_expected_shape(q_, (4,))

        # quaternion values are stored internally with the following conventions:
        # * wxyz-order
        # * w >= 0
        # * norm(q) == 1 (within numerical accuracy)
        if q_order == QuaternionOrder.XYZW:
            q_ = _xyzw_to_wxyz(q_)
        if q_[0] < 0:
            q_ *= -1
        q_norm = np.linalg.norm(q_)
        if np.isclose(q_norm, 0, rtol=0.0, atol=QUATERNION_TOLERANCE):
            raise ValueError(f"Input quaternion has zero length (within tolerance): {q}.")
        self._q: UNIT_QUATERNION_T = q_ / q_norm

    @property
    def _q_conjugate(self):
        return self._q * np.array([1.0, -1.0, -1.0, -1.0])

    def __repr__(self) -> str:
        return "Rotation({:.8f} | {:.8f}, {:.8f}, {:.8f})".format(*self._q)

    @overload
    def __matmul__(self, other: "Rotation") -> "Rotation":
        ...

    # note: imprecise overload for now, is the best we can do for numpy <= 1.19
    @overload
    def __matmul__(
        self, other: Union[ARRAY_LIKE_1D_T, VECTOR_T, HOMOGENEOUS_VECTOR_T]
    ) -> Union[VECTOR_T, HOMOGENEOUS_VECTOR_T]:
        ...

    def __matmul__(
        self, other: Union["Rotation", ARRAY_LIKE_1D_T, VECTOR_T, HOMOGENEOUS_VECTOR_T]
    ) -> Union["Rotation", VECTOR_T, HOMOGENEOUS_VECTOR_T]:

        # concatenation of rotations
        if isinstance(other, self.__class__):
            return self.__class__(self._hamilton_product(self._q, other._q))

        # expect a coordinate vector that we should apply the rotation to
        vector: Union[VECTOR_T, HOMOGENEOUS_VECTOR_T] = np.asarray(other, dtype=np.float64).squeeze()
        self._raise_if_not_expected_shape(vector, {(3,), (4,)})
        is_homogeneous = vector.shape == (4,)
        if is_homogeneous and not np.isclose(vector[3], 1, rtol=0):
            raise ValueError(f"{other} has the shape of a homogeneous vector, but the last component is not 1.")

        q_res = self._hamilton_product(self._hamilton_product(self._q, np.r_[0, vector[:3]]), self._q_conjugate)
        if is_homogeneous:
            return np.r_[q_res[1:], 1.0]
        return q_res[1:]

    def __pow__(self, power: float) -> "Rotation":
        omega = np.arccos(self._q[0])
        if np.isclose(omega, 0) or np.isclose(omega, np.pi):
            # only happens if self._q[0] close to -+ 1, so vector part of quaternion is close to (0, 0, 0)
            v = np.array([0, 0, 0])
        else:
            v = self._q[1:] / np.sin(omega)
        return Rotation(np.r_[np.cos(power * omega), v * np.sin(power * omega)])

    def random(self, random_state) -> "Rotation":
        raise NotImplementedError()

    def slerp(self):
        raise NotImplementedError()

    @staticmethod
    def identity() -> "Rotation":
        return Rotation([1, 0, 0, 0])

    @classmethod
    def from_matrix(cls, matrix: Union[ROTATION_MATRIX_T, HOMOGENEOUS_MATRIX_T, ARRAY_LIKE_2D_T]) -> "Rotation":
        """
        factory to construct a Rotation from matrix input

        uses Shepperd's method:
        S.W. Sheppard, “Quaternion from rotation matrix,” Journal of Guidance and Control,
        Vol. 1, No. 3, pp. 223-224, 1978

        :param matrix: 3x3 rotation matrix or 4x4 homogeneous transformation matrix (of which only the rotation-part
                       will be used)
        """
        # todo: skipping of check for proper rotation matrix via flag?
        # todo: see https://upcommons.upc.edu/bitstream/handle/2117/178326/2083-A-Survey-on-the-Computation-of-Quaternions-from-Rotation-Matrices.pdf
        #       for other, better methods?
        m: np.ndarray = np.asanyarray(matrix, dtype=np.float64).squeeze()
        cls._raise_if_not_expected_shape(m, {(3, 3), (4, 4)})
        if not is_rotation_matrix(m[:3, :3]):
            raise ValueError(f"Input matrix does not describe a proper rotation: {matrix}")

        trace = np.trace(m[:3, :3])
        if trace > 0:
            w = 0.5 * np.sqrt(1 + trace)
            c = 0.25 * w
            return Rotation(
                [
                    w,
                    c * (m[2, 1] - m[1, 2]),
                    c * (m[0, 2] - m[2, 0]),
                    c * (m[1, 0] - m[0, 1]),
                ],
            )

        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            c = 0.5 / np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            return Rotation(
                [
                    c * (m[2, 1] - m[1, 2]),
                    0.25 / c,
                    c * (m[0, 1] + m[1, 0]),
                    c * (m[0, 2] + m[2, 0]),
                ]
            )
        if m[1, 1] > m[2, 2]:
            c = 0.5 / np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2])
            return Rotation(
                [
                    c * (m[0, 2] - m[2, 0]),
                    c * (m[0, 1] + m[1, 0]),
                    0.25 / c,
                    c * (m[1, 2] + m[2, 1]),
                ]
            )

        c = 0.5 / np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2])
        return Rotation(
            [
                c * (m[1, 0] - m[0, 1]),
                c * (m[0, 2] + m[2, 0]),
                c * (m[1, 2] + m[2, 1]),
                0.25 / c,
            ]
        )

    @classmethod
    def from_angle_axis(cls, angle: float, axis: Union[VECTOR_T, ARRAY_LIKE_1D_T]) -> "Rotation":
        """
        factory to construct a rotation form axis-angle input

        :param angle: rotation angle in radian
        :param axis: shape (3,) 3D axis to rotate around, will be normalized internally but must not be of length 0
        """

        axis_: np.ndarray = np.asarray(axis, dtype=np.float64).squeeze()
        cls._raise_if_not_expected_shape(axis_, (3,))

        axis_norm = np.linalg.norm(axis_)
        if np.isclose(axis_norm, 0, rtol=0.0):
            raise ValueError(f"Rotation axis must not be of (close to) zero length. Input: {axis}")

        s = np.sin(angle / 2)
        w = np.cos(angle / 2)
        unit_vector = axis_ / axis_norm
        return Rotation((w, s * unit_vector[0], s * unit_vector[1], s * unit_vector[2]))

    # todo: refactor checks, reoccurring code
    # todo: reject invalid inputs
    # todo: unittests checking individual values
    @classmethod
    def from_rotation_vector(cls, rotation_vector: Union[VECTOR_T, ARRAY_LIKE_1D_T]) -> "Rotation":
        """
        factory to construct a rotation form a rotation vector (sometimes also called Euler vector or
        equal-angle-axis-representation)

        :param rotation_vector: shape (3,) rotation vector
        """
        rvec: np.ndarray = np.asarray(rotation_vector, dtype=np.float64).squeeze()
        cls._raise_if_not_expected_shape(rvec, (3,))

        angle = float(np.linalg.norm(rvec))
        if np.isclose(angle, 0, rtol=0.0):
            return cls.from_angle_axis(angle, [1, 0, 0])
        else:
            return cls.from_angle_axis(angle, rvec)  # axis does not need to be normalized, so a factor of angle is fine

    @classmethod
    def from_euler(cls, euler_angles: Sequence[float], axes: str) -> "Rotation":
        raise NotImplementedError()

    @classmethod
    def from_rpy(cls, rpy: Sequence[float]) -> "Rotation":
        raise NotImplementedError()

    # todo: testing -> matrix shape, homog. matrix last rows, columns, rotation matrix properties, there and back again
    #   check identity, check certain examples
    def as_matrix(self, to_homogeneous_matrix: bool = False) -> Union[ROTATION_MATRIX_T, HOMOGENEOUS_MATRIX_T]:
        # todo: doc
        # see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
        w, x, y, z = self._q

        # shortcut for identity
        if np.isclose(w, 1, rtol=0, atol=QUATERNION_TOLERANCE):
            if to_homogeneous_matrix:
                return np.eye(4)
            else:
                return np.eye(3)

        wx = 2 * w * x
        wy = 2 * w * y
        wz = 2 * w * z
        xx = 2 * x * x
        xy = 2 * x * y
        xz = 2 * x * z
        yy = 2 * y * y
        yz = 2 * y * z
        zz = 2 * z * z
        r_matrix = np.array(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ]
        )

        if not to_homogeneous_matrix:
            return r_matrix
        h_matrix = np.eye(4)
        h_matrix[:3, :3] = r_matrix
        return h_matrix

    def as_quaternion(self, q_order: QuaternionOrder = QuaternionOrder.WXYZ) -> UNIT_QUATERNION_T:
        if q_order == QuaternionOrder.XYZW:
            return _wxyz_to_xyzw(self._q)
        return self._q

    def as_angle_axis(self) -> Tuple[float, UNIT_VECTOR_T]:
        """
        return the scalar rotation angle and 3D rotation axis describing the given rotation

        :return: (angle, axis) with angle in [0, pi] and axis as a unit-vector
        """
        w = self._q[0]
        angle = float(2 * np.arccos(w))
        if np.isclose(w, 1, rtol=0):
            axis = np.array([1, 0, 0], dtype=np.float64)
        else:
            axis = self._q[1:] / np.sqrt(1 - w ** 2)

        return angle, axis

    def as_rotation_vector(self) -> VECTOR_T:
        """
        return the rotation vector representation (sometimes also called Euler vector or equal-angle-axis-representation)
        describing the given rotation
        """
        angle, axis = self.as_angle_axis()
        return angle * axis

    def as_euler(self, axes: str) -> np.ndarray:
        raise NotImplementedError()

    def as_rpy(self) -> np.ndarray:
        raise NotImplementedError()

    def inverse(self) -> "Rotation":
        return Rotation(self._q_conjugate)

    def almost_equal(self, other: "Rotation", eps: float = 1e-6) -> bool:
        """
        check if two Rotation objects represent the same rotation within tolerance
        """
        # Check for cosine similarity between 4D-vectors (and keep in mind that q and -q are the same rotation).
        # This behaves numerically more stable than component-wise comparison.
        # See https://gamedev.stackexchange.com/a/75108 for more info.
        return np.abs(np.dot(self._q, other.as_quaternion())) - 1.0 <= eps

    @staticmethod
    def _raise_if_not_expected_shape(a: np.ndarray, expected: Union[Tuple[int, ...], Set[Tuple[int, ...]]]) -> None:
        if not isinstance(expected, set):
            expected = {expected}
        if a.shape not in expected:
            expected_str = ", ".join(str(tupl) for tupl in sorted(list(expected)))
            raise ValueError(
                f"Bad input shape, expected one of {expected_str} (after squeezing), but got {a.shape} instead."
            )

    @staticmethod
    def _hamilton_product(q1: QUATERNION_T, q2: QUATERNION_T) -> QUATERNION_T:
        r1, v1 = q1[0], q1[1:]
        r2, v2 = q2[0], q2[1:]
        return np.r_[
            (r1 * r2 - v1 @ v2,),  # w-part
            r1 * v2 + r2 * v1 + np.cross(v1, v2),  # xyz-part
        ]
