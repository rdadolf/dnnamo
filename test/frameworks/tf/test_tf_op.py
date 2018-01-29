import unittest
import tensorflow as tf

from dnnamo.framework.tf import TFOp

class TestTFOp(unittest.TestCase):
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

  def test_init_from_op(self):
    natops = self._g.get_operations()
    tfops = [TFOp(natop) for natop in natops]

    # Check types
    a_op = TFOp(self._a)
    assert a_op.optype=='Add', 'TFOp picked up incorrect operation type: '+str(a_op.optype)+' should be '+str(a.type)
    assert a_op.id is not None, 'TFOp has an invalid id: '+str(a_op.id)

    for tfop,natop in zip(tfops,natops):
      assert tfop.optype==natop.type, 'TFOp picked up incorrect operation type: '+str(tfop.optype)+' should be '+str(a.type)
    assert tfop.id is not None, 'TFOp has an invalid id: '+str(tfop.id)

    # Check id uniqueness
    assert len(tfops)==len(set([_.id for _ in tfops])), 'Non-unique names found in tfops: '+','.join([str(_.id) for _ in tfops])

  @unittest.skip('No tests actually implemented.') # FIXME
  def test_params_from_op(self):
    natops = self._g.get_operations()
    tfops = [TFOp(natop) for natop in natops]
    # Check parameters
    # FIXME

  def test_init_from_nodedef(self):
    gdef = self._g.as_graph_def()
    nodes = gdef.node
    tfops = [TFOp(node) for node in nodes]

    # Check types
    for tfop,node in zip(tfops, nodes):
      assert tfop.optype==node.op, 'TFOp picked up incorrect operation type: '+str(tfop.optype)+' should be '+str(node.op)

    # Check id uniqueness
    assert len(tfops)==len(set([_.id for _ in tfops])), 'Non-unique names found in tfops: '+','.join([str(_.id) for _ in tfops])


  @unittest.skip('No tests actually implemented.') # FIXME
  def test_params_from_nodedef(self):
    gdef = self._g.as_graph_def()
    nodes = gdef.node
    tfops = [TFOp(node) for node in nodes]
    # Check parameters
    # FIXME
