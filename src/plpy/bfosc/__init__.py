"""
Data reduction pipelines for 2.16-m/BFOSC.
"""

__all__ = ['g4_old', 'phot_old', 'phot']

from .g4_old import pipeline as g4_old
from .phot_old import pipeline as phot_old
from .phot import pipeline as phot