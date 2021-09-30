"""
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


from hypothesis import assume, example, given, settings
from pytest import mark, raises
from pytf3d import QuaternionOrder, Rotation
from pytf3d.testing import (
    HomogeneousVectorStrategy,
    QuaternionStrategy,
    RotationStrategy,
    UnitQuaternionStrategy,
    VectorStrategy,
)
from pytf3d.typing import (
    ARRAY_LIKE_1D_T,
    HOMOGENEOUS_MATRIX_T,
    HOMOGENEOUS_VECTOR_T,
    QUATERNION_T,
    ROTATION_MATRIX_T,
    VECTOR_T,
)
from pytf3d.utils import is_homogeneous_matrix, is_rotation_matrix
from typing import Any, List, Type, Union

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest


@given(
    valid_quaterion=QuaternionStrategy,
    quat_order_in=st.sampled_from(QuaternionOrder),
    quat_order_out=st.sampled_from(QuaternionOrder),
)
def test_instantiation_with_valid_values(
    valid_quaterion: QUATERNION_T, quat_order_in: QuaternionOrder, quat_order_out: QuaternionOrder
):
    """
    smoke test trying to instantiate a rotation with valid values
    """
    r = Rotation(valid_quaterion, quat_order_in)
    q_out = r.as_quaternion(quat_order_out)

    assert np.isclose(1.0, np.linalg.norm(q_out)), "expected unit quaternion"

    w_idx = -1 if quat_order_out == QuaternionOrder.XYZW else 0
    assert q_out[w_idx] >= 0, "expected positive w-component"


@mark.parametrize(
    ["invalid_value", "expected_error"],
    [
        [(0.0, 0.0, 0.0, 0.0), ValueError],  # zero norm
        [(1.0, 1.0, 1.0, 1.0, 1.0), ValueError],  # too many elements
        [(1.0, 1.0, 1.0), ValueError],  # too few elements
        [[], ValueError],
        [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], ValueError],  # bad shape
    ],
)
@given(
    quat_order_in=st.sampled_from(QuaternionOrder),
)
def test_instantiation_with_invalid_values(
    invalid_value: Any, quat_order_in: QuaternionOrder, expected_error: Type[Exception]
):
    with raises(expected_error):
        _ = Rotation(invalid_value, quat_order_in)


def test_instantiation_with_neg_zero_w():
    q = [-0.0, 1, 0, 0]
    r = Rotation(q)
    q_ret = r.as_quaternion()
    assert all(q_ret == [0, -1, 0, 0])


@given(q=UnitQuaternionStrategy, diff=UnitQuaternionStrategy, diff_norm=st.sampled_from([0, 1e-6]))
def test_equality_returns_true_for_equality(q: QUATERNION_T, diff: QUATERNION_T, diff_norm: float):
    q_plus_diff = q + diff * diff_norm
    assert Rotation(q).almost_equal(Rotation(q_plus_diff))


@given(q1=UnitQuaternionStrategy, q2=UnitQuaternionStrategy)
def test_equality_returns_false_for_inequality(q1: QUATERNION_T, q2: QUATERNION_T):
    assume(not np.allclose(q1, q2, atol=1e-6))
    assume(not np.allclose(q1, -q2, atol=1e-6))
    assert not Rotation(q1).almost_equal(Rotation(q2))


@mark.parametrize(
    ["matrix", "expected"],
    [
        [np.diagflat([-1, -1, 1]), Rotation([0, 0, 0, 1])],
        [np.diagflat([-1, -1, 1]).astype(np.int64), Rotation([0, 0, 0, 1])],
        [np.diagflat([-1, -1, 1]).astype(np.float32), Rotation([0, 0, 0, 1])],
        [np.diagflat([1, 1, 1, 1]), Rotation.identity()],  # homogeneous matrix input
    ],
)
def test_from_matrix_valid_input(matrix: Union[ROTATION_MATRIX_T, HOMOGENEOUS_MATRIX_T], expected: Rotation):
    r = Rotation.from_matrix(matrix)
    assert expected.almost_equal(r)


@given(angle=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False))
def test_from_matrix_Rx(angle: float):
    # fmt: off
    matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]]
    )
    # fmt: on
    q = (np.cos(angle / 2), np.sin(angle / 2), 0, 0)
    assert Rotation(q).almost_equal(Rotation.from_matrix(matrix))


@mark.parametrize(
    ["matrix", "expected_error", "error_regex"],
    [
        [np.diagflat([2, 1, 1]), ValueError, r"does not describe a proper rotation"],
        [np.diagflat([1, 1, 1, 1, 1]), ValueError, r"Bad input shape"],
        [np.diagflat(["a", "b", "c"]), ValueError, r"could not convert .*? to float"],  # not a valid data type
    ],
)
def test_from_matrix_invalid_input(matrix: Any, expected_error: Type[Exception], error_regex: str):
    with pytest.raises(expected_error, match=error_regex):
        _ = Rotation.from_matrix(matrix)


@given(r=RotationStrategy, homogeneous_matrix=st.booleans())
@example(r=Rotation.identity(), homogeneous_matrix=True)
def test_as_matrix(r: Rotation, homogeneous_matrix: bool):
    matrix = r.as_matrix(homogeneous_matrix)
    if homogeneous_matrix:
        assert is_homogeneous_matrix(matrix)
    else:
        assert is_rotation_matrix(matrix)


@given(r=RotationStrategy, homogeneous_matrix=st.booleans())
def test_rotation_matrix_round_trip(r: Rotation, homogeneous_matrix: bool):
    matrix = r.as_matrix(to_homogeneous_matrix=homogeneous_matrix)
    r_restored = Rotation.from_matrix(matrix)
    assert r.almost_equal(r_restored), f"round trip failed, intermediate matrix:\n{matrix}"


@mark.parametrize(
    ["angle", "axis", "expected"],
    [
        [0, [1, 0, 0], Rotation.identity()],
        [np.pi, [1, 0, 0], Rotation([0, 1, 0, 0])],
        [np.pi, [0, 1, 0], Rotation([0, 0, 1, 0])],
        [np.pi, [0, 0, 1], Rotation([0, 0, 0, 1])],
        [-np.pi, [1, 0, 0], Rotation([0, 1, 0, 0])],
        [np.pi, [2, 0, 0], Rotation([0, 1, 0, 0])],
        [np.pi, [0.5, 0, 0], Rotation([0, 1, 0, 0])],
        [3 * np.pi, [1, 0, 0], Rotation([0, 1, 0, 0])],
        [np.pi / 2, [1, 0, 0], Rotation([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])],
    ],
)
def test_from_angle_axis_valid_input(angle: float, axis: ARRAY_LIKE_1D_T, expected: Rotation):
    r = Rotation.from_angle_axis(angle, axis)
    assert expected.almost_equal(r)


@mark.parametrize(
    ["angle", "axis", "expected_error", "error_regex"],
    [
        [1, [1, 0], ValueError, r"Bad input shape"],
        [1, [1, 0, 0, 0], ValueError, r"Bad input shape"],
        [1, [1, 0, 0, 0], ValueError, r"Bad input shape"],
        [1, ["a", "b", "c"], ValueError, r"could not convert .*? to float"],  # not a valid data type
        [1, [0, 0, 0], ValueError, r"zero length"],
        [1, [0, 0, 1e-10], ValueError, r"zero length"],
    ],
)
def test_from_angle_axis_invalid_input(angle: Any, axis: Any, expected_error: Type[Exception], error_regex: str):
    with pytest.raises(expected_error, match=error_regex):
        _ = Rotation.from_angle_axis(angle, axis)


@given(r=RotationStrategy)
@example(r=Rotation.identity())
def test_as_angle_axis(r: Rotation):
    angle, axis = r.as_angle_axis()

    assert isinstance(angle, float)
    assert 0 <= angle <= np.pi

    assert axis.shape == (3,)
    assert np.isclose(np.linalg.norm(axis), 1)


@given(r=RotationStrategy, axis_factor=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False))
def test_angle_axis_round_trip(r: Rotation, axis_factor: float):
    hypothesis.assume(not np.isclose(axis_factor, 0, rtol=0))
    angle, axis = r.as_angle_axis()
    r_restored = Rotation.from_angle_axis(angle, axis_factor * axis)  # scaling axis should give the same result
    assert r.almost_equal(r_restored), f"round trip failed, intermediate representation:\nangle:{angle}\naxis:{axis}"


@mark.parametrize(
    ["rvec", "expected"],
    [
        [[0, 0, 0], Rotation.identity()],
        [[np.pi, 0, 0], Rotation([0, 1, 0, 0])],
        [[np.pi / 2, 0, 0], Rotation([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])],
        [[-np.pi / 2, 0, 0], Rotation([1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0])],
        [[0, np.pi, 0], Rotation([0, 0, 1, 0])],
        [[0, 0, np.pi], Rotation([0, 0, 0, 1])],
    ],
)
def test_from_rotation_vector_valid_input(rvec: ARRAY_LIKE_1D_T, expected: Rotation):
    r = Rotation.from_rotation_vector(rvec)
    assert expected.almost_equal(r)


@mark.parametrize(
    ["rvec", "expected_error", "error_regex"],
    [
        [[1, 0], ValueError, r"Bad input shape"],
        [[1, 0, 0, 0], ValueError, r"Bad input shape"],
        [[[1, 1], [1, 1]], ValueError, r"Bad input shape"],
    ],
)
def test_from_rotation_vector_invalid_input(rvec: Any, expected_error: Type[Exception], error_regex: str):
    with pytest.raises(expected_error, match=error_regex):
        _ = Rotation.from_rotation_vector(rvec)


@given(r=RotationStrategy)
def test_rotation_vector_round_trip(r: Rotation):
    rvec = r.as_rotation_vector()
    r_restored = Rotation.from_rotation_vector(rvec)
    assert r.almost_equal(r_restored), f"round trip failed, intermediate representation:\n{rvec}"


@mark.parametrize(
    ["r", "v", "expected"],
    [
        [Rotation.identity(), [1, 0, 0], [1, 0, 0]],
        [Rotation.from_angle_axis(np.pi / 2, [1, 0, 0]), [1, 0, 0], [1, 0, 0]],
        [Rotation.from_angle_axis(-np.pi / 2, [1, 0, 0]), [1, 0, 0], [1, 0, 0]],
        [Rotation.from_angle_axis(np.pi / 2, [0, 1, 0]), [1, 0, 0], [0, 0, -1]],
        [Rotation.from_angle_axis(np.pi / 2, [0, 0, 1]), [1, 0, 0], [0, 1, 0]],
        [Rotation.from_angle_axis(np.pi / 2, [0, 0, 1]), [1, 1, 1], [-1, 1, 1]],
    ],
)
def test_matmul_vector(r: Rotation, v: ARRAY_LIKE_1D_T, expected: ARRAY_LIKE_1D_T):
    res = r @ v
    assert isinstance(res, np.ndarray)
    assert res.shape == (3,)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)  # type: ignore


@mark.parametrize(
    ["r", "v", "expected"],
    [
        [Rotation.identity(), [1, 0, 0, 1], [1, 0, 0, 1]],
        [Rotation.from_angle_axis(np.pi / 2, [1, 0, 0]), [1, 0, 0, 1], [1, 0, 0, 1]],
        [Rotation.from_angle_axis(-np.pi / 2, [1, 0, 0]), [1, 0, 0, 1], [1, 0, 0, 1]],
        [Rotation.from_angle_axis(np.pi / 2, [0, 1, 0]), [1, 0, 0, 1], [0, 0, -1, 1]],
        [Rotation.from_angle_axis(np.pi / 2, [0, 0, 1]), [1, 0, 0, 1], [0, 1, 0, 1]],
        [Rotation.from_angle_axis(np.pi / 2, [0, 0, 1]), [1, 1, 1, 1], [-1, 1, 1, 1]],
    ],
)
def test_matmul_homogeneous_vector(r: Rotation, v: ARRAY_LIKE_1D_T, expected: ARRAY_LIKE_1D_T):
    res = r @ v
    assert isinstance(res, np.ndarray)
    assert res.shape == (4,)
    assert res[3] == 1
    assert res.dtype == np.float64
    assert np.allclose(res, expected)  # type: ignore


@mark.parametrize(
    ["r", "other", "expected"],
    [
        [Rotation.identity(), Rotation.identity(), Rotation.identity()],
        [Rotation.identity(), Rotation.from_angle_axis(np.pi, [1, 0, 0]), Rotation.from_angle_axis(np.pi, [1, 0, 0])],
        [
            Rotation.from_angle_axis(np.pi / 2, [1, 0, 0]),
            Rotation.from_angle_axis(np.pi / 2, [1, 0, 0]),
            Rotation.from_angle_axis(np.pi, [1, 0, 0]),
        ],
        [
            Rotation.from_angle_axis(np.pi, [1, 0, 0]),
            Rotation.from_angle_axis(-np.pi / 2, [0, 0, 1]),
            Rotation.from_angle_axis(np.pi, [1, 1, 0]),
        ],
    ],
)
def test_matmul_rotation(r: Rotation, other: Rotation, expected: Rotation):
    res = r @ other
    assert res.almost_equal(expected)


@mark.parametrize(
    ["inp", "expected_error", "error_regex"],
    [
        [[0, 0, 0, 2], ValueError, "has the shape of a homogeneous vector, but the last component is not 1."],
        [[0, 0], ValueError, "Bad input shape"],
        [["a", "b", "c"], ValueError, r"could not convert .*? to float"],
    ],
)
@given(
    r=RotationStrategy,
)
@settings(max_examples=10)
def test_matmult_bad_inputs(r: Rotation, inp: Any, expected_error: Type[Exception], error_regex: str):
    with pytest.raises(expected_error, match=error_regex):
        _ = r @ inp


@given(r=RotationStrategy, v=VectorStrategy)
def test_matmul_vector_gives_same_result_as_rotation_matrix(r: Rotation, v: ARRAY_LIKE_1D_T):
    rotation_res = r @ v
    matrix_res = r.as_matrix() @ v
    assert np.allclose(rotation_res, matrix_res, atol=1e-3)


@given(r=RotationStrategy, v=HomogeneousVectorStrategy)
def test_matmul_homogeneous_vector_gives_same_result_as_rotation_matrix(r: Rotation, v: ARRAY_LIKE_1D_T):
    rotation_res = r @ v
    matrix_res = r.as_matrix(to_homogeneous_matrix=True) @ v
    assert np.allclose(rotation_res, matrix_res, atol=1e-3)


@given(r1=RotationStrategy, r2=RotationStrategy, homogeneous=st.booleans())
def test_matmul_rotation_gives_same_result_as_rotation_matrix(r1: Rotation, r2: Rotation, homogeneous: bool):
    rotation_res = r1 @ r2
    matrix_res = r1.as_matrix(homogeneous) @ r2.as_matrix(homogeneous)
    assert np.allclose(rotation_res.as_matrix(homogeneous), matrix_res, atol=1e-3)


@given(r=RotationStrategy)
def test_inverse(r: Rotation):
    ident_1 = r @ r.inverse()
    ident_2 = r.inverse() @ r

    assert Rotation.identity().almost_equal(ident_1)
    assert Rotation.identity().almost_equal(ident_2)


@given(r=RotationStrategy, power=st.integers(0, 20))
def test_positive_int_powers(r: Rotation, power: int):
    r1 = r ** power
    r2 = r.identity()
    for _ in range(power):
        r2 @= r
    assert r1.almost_equal(r2)


@given(r=RotationStrategy, power=st.floats(-20, 20))
def test_power_inverse_relation(r: Rotation, power: float):
    r1 = (r ** power).inverse()
    r2 = r ** -power
    assert r1.almost_equal(r2)


@given(power=st.floats(-20, 20))
def test_identity_powers(power: float):
    ident = Rotation.identity()
    assert ident.almost_equal(ident ** power)


@given(r=RotationStrategy, power=st.floats(-20, 20))
def test_power_rotation_vector_relation(r: Rotation, power: float):
    r_vec = r.as_rotation_vector()
    r1 = r ** power
    r2 = r.from_rotation_vector(power * r_vec)
    assert r1.almost_equal(r2)


@given(r=RotationStrategy, vector=VectorStrategy | HomogeneousVectorStrategy)
def test_matmul_vector_round_trip_via_inverse(r: Rotation, vector: Union[VECTOR_T, HOMOGENEOUS_VECTOR_T]):
    v_rotated = r @ vector
    v_rotated_back = r.inverse() @ v_rotated
    assert np.allclose(vector, v_rotated_back, atol=1e6), f"Vector missmatch, diff: {vector - v_rotated_back}"


@given(r1=RotationStrategy, r2=RotationStrategy)
@example(r1=Rotation([0, 1, 1, 0]), r2=Rotation([0, 1, 1, 0]))  # both are the same -> zero difference
@mark.parametrize(
    ["t_range"],
    [
        [[0, 1]],
        [[0, 0.5, 1]],
        [[0, 0.1, 0.2, 0.4, 0.8, 1]],
    ],
)
def test_slerp_start_end_and_length(r1: Rotation, r2: Rotation, t_range: List[float]):
    # just ensure that we did not set up an unsupported testcase
    assert t_range[0] == 0
    assert t_range[-1] == 1

    # actual test
    slerp_lst = list(r1.slerp(r2, t_range))
    assert len(slerp_lst) == len(t_range)
    assert r1.almost_equal(slerp_lst[0])
    assert r2.almost_equal(slerp_lst[-1])


@given(r=RotationStrategy, t_range=st.lists(st.floats(1e-6, 1e6, allow_nan=False), max_size=100))
def test_slerp_around_identity_gives_same_result_as_power(r: Rotation, t_range: List[float]):
    # Note that we need to use lists (or any other sequence type) to get fixed iteration order and being able to iterate
    # over the input in the for-loop BOTH in the t_range and the slerp object
    slerp = Rotation.identity().slerp(r, t_range)
    for t, r_slerp in zip(t_range, slerp):
        r_from_pow = r ** t
        assert r_slerp.almost_equal(r_from_pow)


@given(r1=RotationStrategy, r2=RotationStrategy, t_range=st.lists(st.floats(1e-6, 1e6, allow_nan=False), max_size=100))
def test_slerp_values(r1: Rotation, r2: Rotation, t_range: List[float]):
    # note: use list as we need to iterate over the t_range multiple times
    slerp_lst = list(r1.slerp(r2, t_range))
    r_diff = r2 @ r1.inverse()
    manual_slerp = [r_diff ** t @ r1 for t in t_range]
    for r_slerp, r_manual in zip(slerp_lst, manual_slerp):
        assert r_slerp.almost_equal(r_manual)


@mark.parametrize(
    ["euler_angles", "sequence", "expected"],
    [
        # single axis rotations
        [(np.pi, 0, 0), "exyz", Rotation.from_angle_axis(np.pi, [1, 0, 0])],
        [(0, np.pi, 0), "exyz", Rotation.from_angle_axis(np.pi, [0, 1, 0])],
        [(0, 0, np.pi), "exyz", Rotation.from_angle_axis(np.pi, [0, 0, 1])],
        [(np.pi, 0, 0), "ixyz", Rotation.from_angle_axis(np.pi, [1, 0, 0])],
        [(0, np.pi, 0), "ixyz", Rotation.from_angle_axis(np.pi, [0, 1, 0])],
        [(0, 0, np.pi), "ixyz", Rotation.from_angle_axis(np.pi, [0, 0, 1])],
        # multi-axis rotations giving identity
        [(np.pi, np.pi, np.pi), "exyz", Rotation.identity()],
        [(np.pi, np.pi, np.pi), "ixyz", Rotation.identity()],
        # arbitrary multi-axis rotations
        [(np.pi / 2, np.pi / 2, 0), "ixyz", Rotation([0.5, 0.5, 0.5, 0.5])],
        [(np.pi / 2, np.pi / 2, 0), "exyz", Rotation([0.5, 0.5, 0.5, -0.5])],
        [(np.pi / 2, 0, np.pi / 2), "ixyz", Rotation([0.5, 0.5, -0.5, 0.5])],
        [(np.pi / 2, 0, np.pi / 2), "exyz", Rotation([0.5, 0.5, 0.5, 0.5])],
        # examples taken from https://quaternions.online/
        [np.deg2rad([30, 45, 60]), "izyx", Rotation([0.822, 0.360, 0.440, 0.022])],
        [np.deg2rad([60, 30, 45]), "iyzx", Rotation([0.723, 0.440, 0.532, 0.022])],
    ],
)
def test_from_euler_examples(euler_angles: List[float], sequence: str, expected: Rotation, eps: float = 1e-3):
    r = Rotation.from_euler(euler_angles, sequence)
    assert r.almost_equal(expected, eps)


@mark.parametrize(
    ["r", "sequence", "expected"],
    [
        # NOTE: test cases are brittle, as Euler angles are not unique
        # # single axis rotations
        [Rotation.from_angle_axis(np.pi, [1, 0, 0]), "exyz", [np.pi, 0, 0]],
        [Rotation.from_angle_axis(np.pi, [0, 1, 0]), "exyz", [np.pi, 0, np.pi]],  # equal to [0, pi, 0]
        [Rotation.from_angle_axis(np.pi, [0, 0, 1]), "exyz", [0, 0, np.pi]],
        [Rotation.from_angle_axis(np.pi, [1, 0, 0]), "ixyz", [np.pi, 0, 0]],
        [Rotation.from_angle_axis(np.pi / 2, [1, 0, 0]), "ixyz", [np.pi / 2, 0, 0]],
        [Rotation.from_angle_axis(np.pi, [0, 1, 0]), "ixyz", [np.pi, 0, np.pi]],  # equal to [0, pi, 0]
        [Rotation.from_angle_axis(np.pi, [0, 0, 1]), "ixyz", [0, 0, np.pi]],
        # identity, no matter what rotation order should give 0-angles
        [Rotation.identity(), "ixyz", [0, 0, 0]],
        [Rotation.identity(), "exyz", [0, 0, 0]],
        [Rotation.identity(), "ezyx", [0, 0, 0]],
        [Rotation.identity(), "exzx", [0, 0, 0]],
        [Rotation.identity(), "eyzx", [0, 0, 0]],
    ],
)
def test_as_euler_examples(r: Rotation, sequence: str, expected: List[float]):
    euler_angles = r.as_euler(sequence)
    assert np.allclose(euler_angles, np.array(expected))


# TODO: generalize sequence / make strategy
@given(r=RotationStrategy)
@example(r=Rotation([1, 0, 0, 0]))
@example(r=Rotation([0.41236532, -0.00000000, -0.00000000, -0.91101858]))
@example(r=Rotation([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]))
@example(r=Rotation([0, 1, 0, 0]))
@example(r=Rotation([0, 0, 1, 0]))
@example(r=Rotation([0, 0, 0, 1]))
@example(r=Rotation([0, 0, 1, 1]))
@mark.parametrize(
    ["sequence"],
    [
        ["exyz"],
        ["ezyx"],
        ["ixyz"],
        ["izyx"],
        ["izyz"],
    ],
)
def test_euler_angles_round_trip(r: Rotation, sequence: str):
    euler_angles = r.as_euler(sequence)
    r_from_angles = Rotation.from_euler(euler_angles, sequence)
    assert r.almost_equal(r_from_angles, eps=1e-4)  # TODO: eps settings vs. numerical issues?
