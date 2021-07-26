"""
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


from hypothesis import example, given
from pytest import mark, raises
from pytf3d import QuaternionOrder, Rotation
from pytf3d.testing import QuaternionStrategy, RotationStrategy
from pytf3d.utils import is_homogeneous_matrix, is_rotation_matrix
from typing import Any, Type

import hypothesis.strategies as st
import numpy as np
import pytest
import re


@given(
    valid_quaterion=QuaternionStrategy,
    quat_order_in=st.sampled_from(QuaternionOrder),
    quat_order_out=st.sampled_from(QuaternionOrder),
)
def test_instantiation_with_valid_values(
    valid_quaterion, quat_order_in: QuaternionOrder, quat_order_out: QuaternionOrder
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
def test_from_matrix_valid_input(matrix: np.ndarray, expected: Rotation):
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
def test_from_matrix_invalid_input(matrix: np.ndarray, expected_error: Type[Exception], error_regex: re.Pattern):
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
def test_from_angle_axis_valid_input(angle: float, axis: np.ndarray, expected: Rotation):
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
def test_from_angle_axis_invalid_input(
    angle: float, axis: np.ndarray, expected_error: Type[Exception], error_regex: re.Pattern
):
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


@given(r=RotationStrategy)
def test_angle_axis_round_trip(r: Rotation):
    angle, axis = r.as_angle_axis()
    r_restored = Rotation.from_angle_axis(angle, axis)
    assert r.almost_equal(r_restored), f"round trip failed, intermediate representation:\nangle:{angle}\naxis:{axis}"
