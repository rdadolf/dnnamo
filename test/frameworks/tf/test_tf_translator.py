import unittest

import tensorflow as tf

from dnnamo.core.model import DynamicModel
from dnnamo.frameworks.tf.tf_translator import TFTranslator

class SyntheticModelSkeleton(DynamicModel):
  def __init__(self):
    self._g = tf.Graph()

  def get_graph(self): return self._g
  def get_weights(self, keys=None): return {}
  def set_weights(self, kv): pass
  def run_train(self, runstep=None, n_steps=1, *args, **kwargs):
    raise NotImplementedError('Cannot run a synthetic test model.')
  def run_inference(self, runstep=None, n_steps=1, *args, **kwargs):
    raise NotImplementedError('Cannot run a synthetic test model.')
  def get_activations(self, runstep=None, *args, **kwargs):
    raise NotImplementedError('Cannot run a synthetic test model.')

class SyntheticModel(object):
  def __init__(self):
    self._m = SyntheticModelSkeleton()
  # Wraps the underlying TF graph context's __enter__ / __exit__
  def __enter__(self):
    self._tf_graph_context = self._m.get_graph().as_default()
    self._tf_graph_context.__enter__()
    return self._m
  def __exit__(self, exc_type, exc_value, traceback):
    return self._tf_graph_context.__exit__(exc_type, exc_value, traceback)

class TestTFTranslator(unittest.TestCase):
  def test_tfop_matmul(self):
    translator = TFTranslator()
    with SyntheticModel() as m:
      t = tf.matmul( tf.constant([1,2],shape=[1,2]), tf.constant([3,4],shape=[2,1]) )
      dg = translator.translate( m )
      assert len(dg)>0, 'No primops in resultant dependence graph.'
      primop_id = translator.map_native_op( t.op.name )

      primop = dg[primop_id]
      assert primop.optype=='mmmul', 'Incorrect primop type for tf.MatMul: '+str(primop)
