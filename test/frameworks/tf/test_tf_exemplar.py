import unittest

from dnnamo.framework.tf.tf_exemplar import *

class TestTFExemplar(unittest.TestCase):
  def test_instantiate_all(self):
    _ = TFExemplar_zero( [] )
    _ = TFExemplar_hadamard( [('dim',[])] )
    _ = TFExemplar_dot( [('dim_a',[]), ('dim_b',[]), ('dim_reduce',1) ] )
    _ = TFExemplar_convolution( [('dim_a',[]), ('dim_b',[])] )
