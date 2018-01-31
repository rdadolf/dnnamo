import unittest

import tensorflow as tf

from dnnamo.core.identifier import OP
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

      # Check on the MatMul op
      # We can do this because of the way TFGraph.from_root_graph preserves names
      t_natop_id = OP(t.op.name)
      t_natop = ng.op(t_natop_id)

      # We can do this because of the way TF translation rules preserve names
      primop_candidates = [_ for _ in pg.ops if _.root.id==t_natop_id]
      assert len(primop_candidates)==1
      t_primop = primop_candidates[0]

      # Checks
      assert t_natop.optype=='MatMul', 'Incorrect native op type (is the test wrong?): '+str(t_natop.optype)
      assert t_primop.optype=='dot', 'Incorrect primitive op type generated from '+str(t_natop.optype)+': '+str(t_primop.optype)

