import unittest
import pytest

from dnnamo.core.primop import PrimopTypes
from dnnamo.core.argsampler import UniformArgSampler

class TestUniformArgSampler(unittest.TestCase):
  def test_instantiate(self):
    UniformArgSampler()

@pytest.mark.parametrize('primop_type', [
  'undef',
  'zero',
  'hadamard',
  #'dot',
  #'convolution'
  ])
class TestUniformArgSamplerOps(object):

  def test_single_sample(self, primop_type):
    sampler = UniformArgSampler()
    args = sampler.sample(primop_type, n=1, seed=1)[0] # index 0
    PType = PrimopTypes.lookup(primop_type)
    primop = PType( args )
    assert len(args)==len(primop.arguments), 'Argument length mismatch'

  def test_multi_sample(self, primop_type):
    N = 1000
    sampler = UniformArgSampler()
    args = sampler.sample(primop_type, n=N, seed=1) # no index
    assert len(args)==N, 'Invalid sample count'
    PType = PrimopTypes.lookup(primop_type)
    for i in xrange(0,N):
      primop = PType( args[i] )
      assert len(args[i])==len(primop.arguments), 'Argument length mismatch'

