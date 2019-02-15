import unittest

import pytest
import tensorflow as tf

from dnnamo.framework.tf import TFGraph

class TestTFGraph(unittest.TestCase):
  @classmethod
  def setUp(self):
    self._g = tf.Graph()
    self._gcontext = self._g.as_default()
    self._gcontext.__enter__()
    self._x = tf.constant(1)
    self._y = tf.constant(2)
    self._z = self._x + self._y
    self._a = self._z.op

  @classmethod
  def tearDown(self):
    # tearDown doesn't pass exceptions, so neither can we
    self._gcontext.__exit__(type=None,value=None,traceback=None)

  @pytest.mark.xfail() # These don't really check anything right now
  def test_init_tfgraph(self):
    TFGraph.from_graph(self._g)
    # FIXME: need tests

  @pytest.mark.xfail() # These don't really check anything right now
  def test_augment_from_rmd(self):
    s = tf.Session()
    opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
    rmd = tf.RunMetadata()
    _ = s.run(self._a, {}, options=opts, run_metadata=rmd)
    TFGraph.from_rmd(rmd)
    # FIXME: need tests

  @pytest.mark.xfail() # NYI
  def test_accessors(self):
    pass
    # g = TFGraph.from_graph(self._g)
    # g.__contains__
    # g.__getitem__
    # g.vertices_from(e)
    # g.vertices_to(e)
    # g.edges_from(v)
    # g.edges_to(v)
    # g.tensors_from(v)
    # g.tensors_to(v)
    # g.dependencies_from(v)
    # g.dependencies_to(v)

  @pytest.mark.xfail() # NYI
  def test_mutators(self):
    pass
    # g = TFGraph()
    # g.add_vertex
    # g.add_edge
