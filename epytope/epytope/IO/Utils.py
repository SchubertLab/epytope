# This code is part of the epytope distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""
.. module:: Utils
   :synopsis: Module provides IO related helper functions
.. moduleauthor:: Leon Kuchenbecker

"""

import contextlib
import sys

@contextlib.contextmanager
def capture_stdout(target):
    """Captures stdout in target temporarily"""
    old = sys.stdout
    try:
        sys.stdout = target
        yield
    finally:
        sys.stdout = old
