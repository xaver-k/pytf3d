"""
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


from pytf3d.typing import HOMOGENEOUS_MATRIX_T, ROTATION_MATRIX_T

import numpy as np


def is_rotation_matrix(r: ROTATION_MATRIX_T) -> bool:
    """
    :return True if the given matrix has all properties of a rotation matrix, else False
    """
    return (
        r.shape == (3, 3)
        and np.isclose(np.linalg.det(r), 1, rtol=0)
        and np.allclose(r @ np.transpose(r), np.identity(3), rtol=0)
    )


def is_homogeneous_matrix(h: HOMOGENEOUS_MATRIX_T) -> bool:
    """
    :return False if the given matrix has all properties of a homogeneous transformation matrix, else False
    """
    return h.shape == (4, 4) and is_rotation_matrix(h[:3, :3]) and np.allclose(h[3, ...], (0, 0, 0, 1), rtol=0)
