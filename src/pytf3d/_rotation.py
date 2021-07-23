from enum import auto, Enum, unique
from pytf3d.typing import (
    ARRAY_LIKE_1D_T,
    ARRAY_LIKE_2D_T,
    HOMOGENEOUS_MATRIX_T,
    QUATERNION_T,
    ROTATION_MATRIX_T,
    UNIT_QUATERNION_T,
)
from pytf3d.utils import is_rotation_matrix
from typing import Sequence, Union

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

        if q_.shape != (4,):
            raise ValueError(f"Not a valid quaternion shape ({q_.shape}) from input: {q}")

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

    def __repr__(self) -> str:
        return "Rotation({:.8f} | {:.8f}, {:.8f}, {:.8f})".format(*self._q)

    def __matmul__(self, other: "Rotation") -> "Rotation":
        raise NotImplementedError()

    def __pow__(self, power, modulo=None) -> "Rotation":
        raise NotImplementedError()

    def random(self, random_state) -> "Rotation":
        raise NotImplementedError()

    def slerp(self):
        raise NotImplementedError()

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
        if m.shape not in {(3, 3), (4, 4)}:
            raise ValueError(f"Bad input shape, expected 3x3 or 4x4 (after squeezing, but gut {m.shape} instead.")
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
    def from_euler(cls, euler_angles: Sequence[float], axes: str) -> "Rotation":
        raise NotImplementedError()

    @classmethod
    def from_rpy(cls, rpy: Sequence[float]) -> "Rotation":
        raise NotImplementedError()

    @classmethod
    def from_eea(cls, eea: Sequence[float]) -> "Rotation":
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

    def as_euler(self, axes: str) -> np.ndarray:
        raise NotImplementedError()

    def as_rpy(self) -> np.ndarray:
        raise NotImplementedError()

    def as_eaa(self) -> np.ndarray:
        raise NotImplementedError()

    def inverse(self) -> "Rotation":
        inverse_q = self._q.copy()
        inverse_q[0] *= -1.0
        return Rotation(inverse_q)

    def almost_equal(self, other: "Rotation") -> bool:
        """
        check if two rotations are equal within tolerance
        """
        # do not need to consider quaternion eqality of q and -q, because constructor ensures that the w-component is
        # always positive
        return np.allclose(self._q, other.as_quaternion())
