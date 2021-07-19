from hypothesis import given
from numpy.testing import assert_raises
from pytest import mark
from pytf3d import QuaternionOrder, Rotation
from pytf3d.testing import QuaternionStrategy
from typing import Any, Type

import hypothesis.strategies as st
import numpy as np


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

    with assert_raises(expected_error):
        _ = Rotation(invalid_value, quat_order_in)
