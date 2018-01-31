import unittest
import tensorflow as tf

from dnnamo.core.identifier import T
from dnnamo.framework.tf import TFTensor

class TestTFTensor(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls._g = tf.Graph()
    cls._gcontext = cls._g.as_default()
    cls._gcontext.__enter__()
    cls._x = tf.constant(1)
    cls._y = tf.constant([2,3])
    cls._z = cls._x + cls._y

  @classmethod
  def tearDown(cls):
    cls._gcontext.__exit__(None,None,None)

  def test_initialization(self):
    t = TFTensor('test',[1,2,3],srcs=[],dsts=[],root=None)
    t = TFTensor('test',[1],srcs=[],dsts=[],root=None)
    t = TFTensor('test',[],srcs=[],dsts=[],root=None)

  def test_from_tensor(self):
    x = TFTensor.from_root_tensor(self._g, self._x)
    y = TFTensor.from_root_tensor(self._g, self._y)
    z = TFTensor.from_root_tensor(self._g, self._z)

    # Check dimensions
    assert x.shape==[]
    assert y.shape==[2]
    assert z.shape==[2]

    # Check connections
    assert x.dsts[0] in z.srcs
    assert y.dsts[0] in z.srcs
