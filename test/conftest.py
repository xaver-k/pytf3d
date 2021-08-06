"""
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hypothesis import settings

import os

# hypothesis run profiles
common = settings(derandomize=True)
settings.register_profile("ci", parent=common, max_examples=1000, print_blob=True)
settings.register_profile("dev", parent=common, max_examples=50)
settings.load_profile(os.getenv(u"HYPOTHESIS_PROFILE", "dev").lower())
