import unittest
import tensorflow as tf

from dnnamo.framework.tf import TFGraph

class TestTFGraph(unittest.TestCase):
  @classmethod
  def setUp(self):
    self._g = tf.Graph()
    self._gcontext = self._g.as_default()
    self._gcontext.__enter__()
    self._x = tf.constant(1)
    self._y = tf.constant(2)
    self._z = self._x + self._y
    self._a = self._z.op

  @classmethod
  def tearDown(self):
    # tearDown doesn't pass exceptions, so neither can we
    self._gcontext.__exit__(type=None,value=None,traceback=None)

  def test_init_tfgraph(self):
    TFGraph.from_graph(self._g)

  def test_consistency(self):
    tfg = TFGraph.from_graph(self._g)
    tfg._check_dataflow_consistency(self._g)


