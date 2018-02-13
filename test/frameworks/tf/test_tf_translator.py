import unittest

import tensorflow as tf

from dnnamo.core.identifier import ID
from dnnamo.core.model import DnnamoModel
from dnnamo.framework.tf.tf_graph import TFGraph
from dnnamo.framework.tf.tf_translator import TFTranslator

class SyntheticModelSkeleton(DnnamoModel):
  def __init__(self):
    self._g = tf.Graph()

  def get_training_graph(self): return self._g
  def get_inference_graph(self): return self._g

class SyntheticModel(object):
  def __init__(self):
    self._m = SyntheticModelSkeleton()
  # Wraps the underlying TF graph context's __enter__ / __exit__
  def __enter__(self):
    self._tf_graph_context = self._m._g.as_default()
    self._tf_graph_context.__enter__()
    return self._m
  def __exit__(self, exc_type, exc_value, traceback):
    return self._tf_graph_context.__exit__(exc_type, exc_value, traceback)

class TestTFTranslator(unittest.TestCase):
  def test_tfop_matmul(self):
    translator = TFTranslator()
    with SyntheticModel() as m:
      t = tf.matmul( tf.constant([1,2],shape=[1,2]), tf.constant([3,4],shape=[2,1]) )
      ng = TFGraph.from_graph(m.get_training_graph())
      assert len(ng.ops)>0, 'Native graph created incorrectly.'

      pg = translator.translate( ng )
      assert len(pg.ops)>0, 'No primops in resultant dependence graph.'

    # XXX Add more checks?
