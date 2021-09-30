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
from typing import Generator, Iterable, overload, Sequence, Set, Tuple, Union

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
        # * w >= +0
        # * norm(q) == 1 (within numerical accuracy)
        if q_order == QuaternionOrder.XYZW:
            q_ = _xyzw_to_wxyz(q_)

        # ensure w > -0 (note the minus sign!)
        q_ *= np.copysign(1.0, q_[0])

        q_norm = np.linalg.norm(q_)
        if np.isclose(q_norm, 0, rtol=0.0, atol=QUATERNION_TOLERANCE):
            raise ValueError(f"Input quaternion has zero length (within tolerance): {q}.")
        self._q: UNIT_QUATERNION_T = q_ / q_norm

    @property
    def _q_conjugate(self):
        return self._q * np.array([-1.0, 1.0, 1.0, 1.0])

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
        """
        multiply the rotation as if it was a rotation matrix, i.e. gives the same result as `r.as_matrix() @ other`, but
        uses faster implementations

        :param other: 3D coordinate vector, homogenous coordinate vector or Rotation
        """

        # concatenation of rotations
        if isinstance(other, self.__class__):
            return self.__class__(self._hamilton_product(self._q, other._q))

        # expect a coordinate vector that we should apply the rotation to
        vector: Union[VECTOR_T, HOMOGENEOUS_VECTOR_T] = np.asarray(other, dtype=np.float64).squeeze()
        self._raise_if_not_expected_shape(vector, {(3,), (4,)})
        is_homogeneous = vector.shape == (4,)
        if is_homogeneous and not np.isclose(vector[3], 1, rtol=0):
            raise ValueError(f"{other} has the shape of a homogeneous vector, but the last component is not 1.")

        # fast vector rotation algorithm, see
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Vector_rotation
        t = 2 * np.cross(self._q[1:], vector[:3])
        v_res = vector[:3] + self._q[0] * t + np.cross(self._q[1:], t)
        if is_homogeneous:
            return np.r_[v_res, 1.0]
        return v_res

    def __pow__(self, power: float) -> "Rotation":
        """
        apply power with the same semantics as if it was a rotation matrix, effectively scaling the rotation
        `power`-times where `power` can be fractional and negative (in which case the rotation is inverted and scaled)
        """
        omega = np.arccos(self._q[0])
        if np.isclose(omega, 0, atol=1e-10) or np.isclose(omega, np.pi, atol=1e-10):
            # only happens if self._q[0] close to -+ 1, so vector part of quaternion is close to (0, 0, 0)
            v = np.array([0, 0, 0])
        else:
            v = self._q[1:] / np.sin(omega)
        return Rotation(np.r_[np.cos(power * omega), v * np.sin(power * omega)])

    def random(self, random_state) -> "Rotation":
        raise NotImplementedError()

    def slerp(self, other: "Rotation", t_range: Iterable[float]) -> Generator["Rotation", None, None]:
        """
        spherical linear interpolation between this Rotation and an other Rotation (using the shortest path) at
        the given values in t_range

        :param other: "endpoint" of interpolation (see also t_range)
        :param t_range: values at which to interpolate, typically in the range of [0, 1] where 0 produces this Rotation
                        and 1 produces other Rotation, but values < 0 and > 1 are also valid for extrapolation
        :return:
        """

        d = self._q @ other._q  # pylint: disable=protected-access
        theta = np.arccos(np.abs(np.clip(d, -1, 1)))  # in 0, pi
        sin_theta = np.sin(theta)

        # special case where interpolation does not make sense
        if np.isclose(sin_theta, 0, atol=1e-10):
            for _ in t_range:
                yield Rotation(self._q)
            return

        # ensure shortest way interpolation by flipping sign for negative d (including d = -0)
        q_self_for_calc = self._q * np.copysign(1.0, d)
        for t in t_range:
            # pylint: disable=protected-access
            q_slerp = (np.sin((1 - t) * theta) * q_self_for_calc + np.sin(t * theta) * other._q) / sin_theta
            yield Rotation(q_slerp)

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
        # todo: skipping of check for proper rotation matrix via flag, as it is expensive?
        m: np.ndarray = np.asanyarray(matrix, dtype=np.float64).squeeze()
        cls._raise_if_not_expected_shape(m, {(3, 3), (4, 4)})
        if not is_rotation_matrix(m[:3, :3]):
            raise ValueError(f"Input matrix does not describe a proper rotation: {matrix}")

        trace = np.trace(m[:3, :3])
        if trace > 0:
            w = 0.5 * np.sqrt(1 + trace)
            c = 0.25 / w
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
        if np.isclose(axis_norm, 0, rtol=0.0, atol=1e-10):
            raise ValueError(f"Rotation axis must not be of (close to) zero length. Input: {axis}")

        s = np.sin(angle / 2)
        w = np.cos(angle / 2)
        unit_vector = axis_ / axis_norm
        return Rotation((w, s * unit_vector[0], s * unit_vector[1], s * unit_vector[2]))

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
        if np.isclose(angle, 0, rtol=0.0, atol=1e-8):
            return cls.identity()
        else:
            return cls.from_angle_axis(angle, rvec)  # axis does not need to be normalized, so a factor of angle is fine

    @classmethod
    def from_euler(cls, angles: Union[VECTOR_T, ARRAY_LIKE_1D_T], sequence: str) -> "Rotation":
        angles_: VECTOR_T = np.asarray(angles, dtype=np.float64).squeeze()
        cls._raise_if_not_expected_shape(angles_, (3,))
        cls._raise_for_invalid_euler_angle_sequence(sequence)
        is_intrinsic_rotation = sequence[0] == "i"
        if is_intrinsic_rotation:
            rotation_seq = sequence[1:]
        else:
            rotation_seq = sequence[-1:0:-1]
            angles_ = angles_[::-1]

        AXES = {
            "x": [1, 0, 0],
            "y": [0, 1, 0],
            "z": [0, 0, 1],
        }

        # TODO: implement more efficiently?
        return (
            cls.from_angle_axis(angles_[0], AXES[rotation_seq[0]])
            @ cls.from_angle_axis(angles_[1], AXES[rotation_seq[1]])
            @ cls.from_angle_axis(angles_[2], AXES[rotation_seq[2]])
        )

    @classmethod
    def from_rpy(cls, rpy: Sequence[float]) -> "Rotation":
        raise NotImplementedError()

    def as_matrix(self, to_homogeneous_matrix: bool = False) -> Union[ROTATION_MATRIX_T, HOMOGENEOUS_MATRIX_T]:
        """
        return the matrix representation of this Rotation

        :param to_homogeneous_matrix: if True, return a homogeneous 4x4 rotation matrix, if False return a "normal"
                                      3x3 rotation matrix
        """
        w, x, y, z = self._q

        # shortcut for identity
        if np.isclose(w, 1, rtol=0, atol=QUATERNION_TOLERANCE):
            if to_homogeneous_matrix:
                return np.eye(4)
            else:
                return np.eye(3)

        # normal conversion path,
        # see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
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
        if np.isclose(w, 1, rtol=0, atol=1e-12):
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

    # TODO: document rotation sequence string
    def as_euler(self, sequence: str) -> np.ndarray:
        """
        return the Euler angles / Tait-Bryan angles that describe the given rotation

        :param sequence: desired rotation sequence for the angles
        :return: angles corresponding to the axes given in the `sequence`-parameter, e.g.:
                 * if sequence="ixzy", will return [angle_x, angle_z, angle_y]
                 * if sequence="ezxy", will return [angle_z, angle_x, angle_y]
                 all angles are guaranteed to be in the range (-pi, pi)
        """

        # algorithm from:
        # Shuster, Malcolm & Markley, Landis. (2006).
        # General Formula for Extracting the Euler Angles.
        # Journal of Guidance Control and Dynamics - 29. 215-221.
        # DOI:10.2514/1.16622.
        # https://www.researchgate.net/publication/238189035_General_Formula_for_Extracting_the_Euler_Angles

        # todo: remove code duplication for axis extraction
        AXES = {
            "x": [1, 0, 0],
            "y": [0, 1, 0],
            "z": [0, 0, 1],
        }

        self._raise_for_invalid_euler_angle_sequence(sequence)
        intrinsic = sequence[0] == "i"

        # the below algorithm assumes intrinsic rotation order, invert order for extrinsic rotations
        multiplication_sequence = sequence[1:] if intrinsic else sequence[-1:0:-1]
        axes = np.array([AXES[axis_label] for axis_label in multiplication_sequence])

        D = self.as_matrix().transpose()  # because paper uses transposed matrices

        # eq. (5), because we use only orthonormal axes, possible values are [-pi/2, 0, pi/2, pi]
        lambd = np.arctan2(np.dot(np.cross(axes[0], axes[1]), axes[2]), np.dot(axes[0], axes[2]))

        # eq. (6), due to row first ordering, the array is already transposed
        C = np.array([axes[1], np.cross(axes[0], axes[1]), axes[0]])

        # eq. (7)
        R_transposed = Rotation.from_angle_axis(lambd, [1, 0, 0]).as_matrix()
        O = R_transposed @ C @ D @ C.transpose()

        # eq. (10a)
        theta = lambd + np.arccos(np.clip(O[2, 2], -1, 1))

        # conditions for eq. (10b) and (10c)
        eps = 1e-6
        cond_1 = np.abs(theta - lambd) >= eps
        cond_2 = np.abs(theta - lambd - np.pi) >= eps

        if cond_1 and cond_2:
            # good observability
            # eq. (10a) and (10b)
            phi = np.arctan2(O[2, 0], -O[2, 1])
            psi = np.arctan2(O[0, 2], O[1, 2])
        else:
            # gimbal lock, set psi to 0
            psi = 0
            if not cond_1:
                # equation (11a)
                phi = np.arctan2(O[0, 1] - O[1, 0], O[0, 0] + O[1, 1])
            else:
                # equation (11b)
                phi = np.arctan2(O[0, 1] + O[1, 0], O[0, 0] - O[1, 1])

        # handle edge-cases of theta range
        if np.isclose(theta, 0):
            theta = 0
        if np.isclose(theta, np.pi):
            theta = np.pi

        # adjust angle ranges if possible
        if not np.isclose(lambd, 0) and not 0 <= theta < np.pi:
            angles = np.mod(np.array([phi + np.pi, 2 * lambd - theta, psi - np.pi]), (2 * np.pi))
            # modulo operations do not catch values close to 2 pi
            close_to_2_pi = np.logical_or(np.isclose(angles, -2 * np.pi), np.isclose(angles, 2 * np.pi))
            angles[close_to_2_pi] = 0.0
        else:
            angles = np.array([phi, theta, psi])

        # ensure range -pi ... pi
        angles[angles < np.pi] += 2 * np.pi
        angles[angles > np.pi] -= 2 * np.pi

        if intrinsic:
            return angles
        else:
            # reverse returned angles for extrinsic rotation
            return angles[::-1]

    def as_rpy(self) -> np.ndarray:
        raise NotImplementedError()

    def inverse(self) -> "Rotation":
        """
        return a new Rotation object, representing the inverse rotation of this Rotation
        """
        return Rotation(self._q_conjugate)

    def almost_equal(self, other: "Rotation", eps: float = 1e-6) -> bool:
        """
        check if two Rotation objects represent the same rotation within tolerance

        :param other: rotation to check against
        :param eps: comparison tolerance, note that we compare the cosine tolerance and use eps**2 internally for that
        """
        # Check for cosine similarity between 4D-vectors (and keep in mind that q and -q are the same rotation).
        # This behaves numerically more stable than component-wise comparison.
        # See https://gamedev.stackexchange.com/a/75108 for more info.
        cosine_similarity = np.abs(np.dot(self._q, other.as_quaternion()))
        return float(cosine_similarity) > 1.0 - eps ** 2

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

    @classmethod
    def _raise_for_invalid_euler_angle_sequence(cls, sequence: str) -> None:
        VALID_REFERENCE_FRAMES = {*"ei"}
        VALID_AXES = {*"xyz"}

        if len(sequence) != 4:
            raise ValueError(f"Invalid euler angle sequence '{sequence}', must be exactly 4 characters.")
        if sequence[0] not in VALID_REFERENCE_FRAMES:
            raise ValueError(
                f"Invalid reference frame specifier for euler angle sequence at position 0: '{sequence}', "
                f"must be either 'e' for extrinsic or 'i' for intrinsic rotations."
            )
        invalid_axes = {*sequence[1:]}.difference(VALID_AXES)
        if invalid_axes:
            raise ValueError(
                f"Invalid axis specifier(s) for euler angle sequence: {invalid_axes}, must be one of {VALID_AXES}."
            )
        if sequence[1] == sequence[2] or sequence[2] == sequence[3]:
            # todo: do we need to check for this? It is only a "courtesy", the rest of the code will still work, even
            #   if this condition is not met
            raise ValueError(f"Invalid axis sequence '{sequence}', consecutive axes must be different.")
