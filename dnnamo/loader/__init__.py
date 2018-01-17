# This is a meta import file which collects loaders from all over

from .runpy_loader import RunpyLoader

from ..framework.tf.loader import *
from ..framework.tf.loader import __all__ as tf_all

__all__ = ['RunpyLoader'] + tf_all
