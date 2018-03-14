import unittest

from dnnamo.framework.tf.tf_synthesis import TFSyntheticModel
from dnnamo.framework.tf.tf_exemplar import TFExemplar_hadamard
from dnnamo.framework import FRAMEWORKS

class TestTFSyntheticModel(unittest.TestCase):
  def setUp(self):
    self.ex = TFExemplar_hadamard([1,2,3,0])

  def test_instantiation(self):
    TFSyntheticModel( self.ex )

  def test_run_it(self):
    m = TFSyntheticModel( self.ex )

    _ = m.run_inference()
    _ = m.profile_inference()

  def test_exemplar_name(self):
    m = TFSyntheticModel( self.ex )
    name = m.get_exemplar_op_name()
    assert name is not None

  def test_framework_interface(self):
    frame = FRAMEWORKS['tf']()
    frame.set_model( TFSyntheticModel(self.ex) )

    _ = frame.get_graph(mode='inference', scope='static', ops='native')
    _ = frame.get_graph(mode='inference', scope='dynamic', ops='native')
    _ = frame.get_graph(mode='inference', scope='static', ops='primitive')
    _ = frame.get_graph(mode='inference', scope='dynamic', ops='primitive')

    _ = frame.get_timing(mode='inference', ops='native')
    _ = frame.get_timing(mode='inference', ops='primitive')
