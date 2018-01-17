import unittest
from dnnamo.core.model import DnnamoModel

from dnnamo.loader import RunpyLoader

class TestCoreLoaders(unittest.TestCase):
  _names = [
    'test/test_models/empty_models.py',
    'test/test_models/simple_nnet.py',
    'test/../test/test_models/simple_nnet.py',
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
        assert isinstance(m,DnnamoModel), 'loader returned a non-model object: '+str(m)
        assert m.is_dnnamo_model, 'loader returned a non-model object: '+str(m)
