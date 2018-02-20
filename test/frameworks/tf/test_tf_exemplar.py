import unittest

from dnnamo.framework.tf.tf_exemplar import *

class TestTFExemplar(unittest.TestCase):
  def test_instantiate_all(self):
    _ = TFExemplar_zero()
    _ = TFExemplar_hadamard()
    _ = TFExemplar_dot()
    _ = TFExemplar_convolution()
