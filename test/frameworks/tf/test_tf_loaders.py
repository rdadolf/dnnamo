import unittest
from dnnamo.frameworks.tf import TFFramework
from dnnamo.frameworks.tf.loaders import *
from dnnamo.core.model import BaseModel

from dnnamo.loaders import RunpyLoader

class TestTFLoaders(unittest.TestCase):
  _names = [
    'test/test_models/empty_models',
    'test/test_models/empty_models',
  ]
  _loaders = [
    RunpyLoader,
  ]

  def test_instantiation(self):
    for loader in self._loaders:
      for name in self._names:
        loader(name)

  def test_load(self):
    for loader in self._loaders:
      for name in self._names:
        m = loader(name).load()
        assert isinstance(m,BaseModel), 'loader returned a non-model object: '+str(m)

  @unittest.SkipTest
  def test_load_with_pypath(self):
    # FIXME: NYI
    pass
