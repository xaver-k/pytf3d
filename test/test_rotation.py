"""
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


from hypothesis import example, given, settings
from pytest import mark, raises
from pytf3d import QuaternionOrder, Rotation
from pytf3d.testing import HomogeneousVectorStrategy, QuaternionStrategy, RotationStrategy, VectorStrategy
from pytf3d.typing import ARRAY_LIKE_1D_T, HOMOGENEOUS_MATRIX_T, QUATERNION_T, ROTATION_MATRIX_T
from pytf3d.utils import is_homogeneous_matrix, is_rotation_matrix
from typing import Any, Type, Union

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


@given(r=RotationStrategy, axis_factor=st.floats(allow_nan=False, allow_infinity=False))
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
