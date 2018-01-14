# This is a meta import file which collects loaders from all over

from .runpy_loader import RunpyLoader

from ..frameworks.tf.loader import *
from ..frameworks.tf.loader import __all__ as tf_all

__all__ = ['RunpyLoader'] + tf_all
