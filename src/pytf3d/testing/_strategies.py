"""
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hypothesis.extra.numpy import arrays
from pytf3d import QUATERNION_TOLERANCE, Rotation
from typing import Sequence

import hypothesis.strategies as st
import numpy as np


def _is_not_almost_norm_0(q: Sequence[float]):
    norm = np.linalg.norm(q)
    return not np.isclose(norm, 0, rtol=0.0, atol=QUATERNION_TOLERANCE)


def _normalize(v: Sequence[float]) -> np.ndarray:
    v_array = np.asarray(v, dtype=float)
    return v_array / np.linalg.norm(v_array)


_quat_entry = st.floats(-1, 1, allow_nan=False, allow_infinity=False)
QuaternionStrategy = st.tuples(_quat_entry, _quat_entry, _quat_entry, _quat_entry).filter(_is_not_almost_norm_0)


UnitQuaternionStrategy = st.builds(_normalize, QuaternionStrategy)
UnitQuaternionStrategy.__doc__ = (
    "Hypothesis strategy producing normalized quaternions (representing rotations) as numpy float arrays"
)

RotationStrategy = st.builds(Rotation, QuaternionStrategy)


def _set_last_value_to_1(a: np.ndarray):
    a[-1] = 1
    return a


# note: +-1e12 is a reasonable range for all coordinate values that we can reasonably expect: picometer -> terrameter
VectorStrategy = arrays(np.float64, (3,), elements=st.floats(-1e-12, 1e12, allow_nan=False))
HomogeneousVectorStrategy = arrays(np.float64, (4,), elements=st.floats(-1e-12, 1e12, allow_nan=False)).map(
    _set_last_value_to_1
)
