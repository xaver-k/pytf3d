"""
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


from typing import Iterable, Union

import numpy as np

# ArrayLike-types, data structure that can be converted to a numpy array
# As we support Python 3.6 and therefore must support numpy 1.19, we need to roll our own.
ARRAY_LIKE_1D_T = Union[Iterable[Union[float, int]], np.ndarray]
ARRAY_LIKE_2D_T = Union[Iterable[Union[ARRAY_LIKE_1D_T, np.ndarray]], np.ndarray]

# misc vectors
VECTOR_T = np.ndarray  # shape (3, ), a vector of 3D-coordinates, such as translations, axes etc
UNIT_VECTOR_T = np.ndarray  # like VECTOR_T, but normalized to length 1
HOMOGENEOUS_VECTOR_T = np.ndarray  # shape (4, ), a homogeneous vector of 3D-coordinates with 1 as the last component

# type definitions for more meaningful type-hints
# Note that these are just aliases right now. With numpy >= 1.21 using numpy.typing.NDArray will be available
# to constrain the types a little more.
QUATERNION_T = np.ndarray  # shape (4,) possibly non-normalized vector representing a quaternion
UNIT_QUATERNION_T = np.ndarray  # shape (4,) normalized unit-quaternion representing a rotation
ROTATION_MATRIX_T = np.ndarray  # shape (3,3) matrix representing rotation
HOMOGENEOUS_MATRIX_T = (
    np.ndarray
)  # shape (4,4) matrix representing a ridgid body transformation (rotation and translation)
