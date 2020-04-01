'''
Created on 7 Nov 2017

@author: szu004
'''
# Ensure backwards compatibility with Python 2
from __future__ import (
    absolute_import,
    division,
    print_function)

from .context import init
from .methods import *

__all__ = ['init', 'random_forest_model']
