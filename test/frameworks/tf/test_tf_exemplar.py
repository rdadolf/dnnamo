import pytest
import unittest

from dnnamo.framework.tf.tf_exemplar import *

@pytest.mark.parametrize('exemplar_pair', [
  #(TFExemplar_zero, []),
  (TFExemplar_hadamard, [1,2,3,4]),
  #(TFExemplar_dot, [('dim_a',[1,2]), ('dim_b',[2,3]), ('dim_reduce',1)]),
  #(TFExemplar_convolution, [('dim_a',[9,9]), ('dim_b', [3,3])]),
])
class TestTFExemplar(object):
  def test_instantiate_all(self, exemplar_pair):
    Exemplar, args = exemplar_pair
    _ = Exemplar(args)

  def test_signatures(self, exemplar_pair):
    Exemplar, args = exemplar_pair
    ex = Exemplar(args)
    isig = ex.input_signature
    assert isig is not None, 'No input signature'
    assert len(isig)>=0, 'Input signature list is not iterable'
    for sig in isig:
      assert type(sig) in TFSignatureType.types, 'Corrupt input signature element'
    osig = ex.output_signature
    assert osig is not None, 'No output signature'
    assert len(osig)>=0, 'Output signature list is not iterable'
    for sig in osig:
      assert type(sig) in TFSignatureType.types, 'Corrupt output signature element'

  def test_synthesize(self, exemplar_pair):
    Exemplar, args = exemplar_pair
    ex = Exemplar(args)

    inputs = []
    for sig in ex.input_signature:
      if isinstance(sig, TFSignatureType.Tensor):
        i = tf.constant( 0, dtype=sig.dtype, shape=sig.dims )
        inputs.append(i)

    outputs = ex.synthesize(inputs)

    for out in outputs:
      assert isinstance(out, tf.Tensor)

