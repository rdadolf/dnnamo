import unittest

import tensorflow as tf

from dnnamo.frameworks.tf.tf_model import TFModel
from dnnamo.frameworks.tf.tf_translator import TFTranslator

class SyntheticTFModel(TFModel):
  def __init__(self):
    self._model = tf.Graph()

  def __enter__(self):
    self._tf_graph_context = self._model.as_default()
    self._tf_graph_context.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self._tf_graph_context.__exit__(exc_type, exc_value, traceback)

  def model(self):
    return self._model

  def run(self, runstep=None, n_steps=1, *args, **kwargs):
    raise NotImplementedError('Cannot run a synthetic test model.')



def genmodel(optype):
  # pylint: disable=unused-variable
  m = SyntheticTFModel()

class TestTFTranslator(unittest.TestCase):

  def test_tfop_matmul(self):
    xlate = TFTranslator()
    with SyntheticTFModel() as m:
      t = tf.matmul( tf.constant([1,2],shape=[1,2]), tf.constant([3,4],shape=[2,1]) )
      dg = xlate.translate( m )
      assert len(dg)>0, 'No primops in resultant dependence graph.'
      primop_id = xlate.map_native_op( t.op.name )

      primop = dg[primop_id]
      assert primop.optype=='mmmul', 'Incorrect primop type for tf.MatMul: '+str(primop)
