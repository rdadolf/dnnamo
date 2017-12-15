import unittest

import numpy as np

from dnnamo.core.stats import NativeStats
from dnnamo.frameworks.tf import TFFramework
from dnnamo.frameworks.tf.tf_stats import TFNativeStats, copy_tf_graph

class TestCopyTFGraph(unittest.TestCase):
  def test_copy(self):
    frame = TFFramework()
    frame.load('test/examples/simple_nnet.py')
    g = frame.native_model()

    h = copy_tf_graph(g)

    # Make sure we actually have a copy
    assert hash(g)!=hash(h), 'Graphs are identical'

    # Check general size
    assert len(h.get_operations())>=len(g.get_operations()), 'Missing ops in graph copy'
    assert len(h.get_operations())<=len(g.get_operations()), 'Extra ops in graph copy'

    # Check op-by-op...
    missing_ops = []
    for gop in g.get_operations():
      try:
        hop = h.get_operation_by_name(gop.name)
      except Exception:
        missing_ops.append(gop.name)
    assert len(missing_ops)<1, str(len(missing_ops))+' missing ops: '+str(missing_ops)
    # ...both ways.
    missing_ops = []
    for hop in h.get_operations():
      try:
        gop = g.get_operation_by_name(hop.name)
      except Exception:
        missing_ops.append(hop.name)
    assert len(missing_ops)<1, str(len(missing_ops))+' missing ops: '+str(missing_ops)


class TestTFStats(unittest.TestCase):
  def test_get_stats_from_framework(self):
    frame = TFFramework()
    frame.load('test/examples/simple_nnet.py')
    stats = frame.native_stats()
    assert stats is not None, 'Failed to get stats from framework'
    assert isinstance(stats, NativeStats), 'Got a bad return value from framework.native_stats(): '+str(type(stats))
    assert isinstance(stats, TFNativeStats), 'Got a bad return value from framework.native_stats(): '+str(type(stats))

    for attr in ['_model','_traces','_meantrace','_flops','_params','_tensors','_src_tensormap','_dst_tensormap']:
      assert hasattr(stats,attr), '_model not initialized in Stats'
      assert getattr(stats,attr) is not None, '_model not initialized in Stats'

  def test_computational_density(self):
    frame = TFFramework()
    frame.load('test/examples/simple_nnet.py')
    stats = frame.native_stats()
    g = frame.native_model()

    # Make sure *everything* gives numerical values.
    for op in g.get_operations():
      flops,bytes = stats.computational_density(op.name)
      assert flops>=0, 'Bad flops value for op "'+str(op.name)+'": '+str(flops)
      assert bytes>=0, 'Bad bytes value for op "'+str(op.name)+'": '+str(bytes)

  #@unittest.SkipTest
  def test_computational_density_values(self):
    frame = TFFramework()
    frame.load('test/examples/simple_nnet.py')
    stats = frame.native_stats()
    #g = frame.native_model()

    # Let's make sure the ops are reporting the right numbers.
    in_size = np.prod(frame.model().input.get_shape().as_list())
    assert in_size == 3200, 'test/examples/simple_nnet has changed since this test case was written; it needs to be updated (and stop changing test files, make your own!)'
    lab_size = np.prod(frame.model().labels.get_shape().as_list())
    assert lab_size==320, 'test/examples/simple_nnet has changed since this test case was written; it needs to be updated (and stop changing test files, make your own!)'

    f32 = 4 # bytes
    for name,ref_flops,ref_bytes in [
        ('matmul', 2*32*100*10, (3200+1000)*f32), # TF's flop estimate is rough
        ('sum', 0, (320+10)*f32), # TF's flop estimate is wrong for sum
        ('inference', 0, 320*f32),
        ]:
      flops,bytes = stats.computational_density(name)
      assert flops==ref_flops, 'Flops are incorect for '+str(name)+': '+str(flops)+' vs '+str(ref_flops)+'(ref)'
      assert bytes==ref_bytes, 'Bytes are incorect for '+str(name)+': '+str(bytes)+' vs '+str(ref_bytes)+'(ref)'

  #@unittest.SkipTest
  def test_computational_density_partial(self):
    frame = TFFramework()
    frame.load('test/examples/placeholder_nnet.py')
    stats = frame.native_stats()
    #g = frame.native_model()

    # Let's make sure the ops are reporting the right numbers.
    #in_size = 3200
    #lab_size = 320

    f32 = 4 # bytes
    for name,ref_flops,ref_bytes in [
        ('matmul', 2*32*100*10, (3200+1000)*f32), # TF's flop estimate is rough
        ('sum', 0, (320+10)*f32), # TF's flop estimate is wrong for sum
        ('inference', 0, 320*f32),
        ]:
      flops,bytes = stats.computational_density(name)
      assert flops==ref_flops, 'Flops are incorect for '+str(name)+': '+str(flops)+' vs '+str(ref_flops)+'(ref)'
      assert bytes==ref_bytes, 'Bytes are incorect for '+str(name)+': '+str(bytes)+' vs '+str(ref_bytes)+'(ref)'
