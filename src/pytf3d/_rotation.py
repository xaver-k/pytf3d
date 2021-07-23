from enum import auto, Enum, unique
from pytf3d.typing import ARRAY_LIKE_1D_T, QUATERNION_T, UNIT_QUATERNION_T
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
    def from_matrix(cls, matrix: np.ndarray) -> "Rotation":
        raise NotImplementedError()

    @classmethod
    def from_euler(cls, euler_angles: Sequence[float], axes: str) -> "Rotation":
        raise NotImplementedError()

    @classmethod
    def from_rpy(cls, rpy: Sequence[float]) -> "Rotation":
        raise NotImplementedError()

    @classmethod
    def from_eea(cls, eea: Sequence[float]) -> "Rotation":
        raise NotImplementedError()

    def as_matrix(self, to_homogeneous_matrix: bool = False) -> np.ndarray:
        raise NotImplementedError()

    def as_quaternion(self, q_order: QuaternionOrder = QuaternionOrder.WXYZ) -> np.ndarray:
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
