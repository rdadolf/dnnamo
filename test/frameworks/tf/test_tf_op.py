import unittest
import tensorflow as tf

from dnnamo.core.identifier import T
from dnnamo.framework.tf import TFOp

from ...test_models.simple_nnet import SimpleNNet

class TestTFOp(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls._g = tf.Graph()
    cls._gcontext = cls._g.as_default()
    cls._gcontext.__enter__()
    cls._x = tf.constant(1)
    cls._y = tf.constant(2)
    cls._z = cls._x + cls._y
    cls._a = cls._z.op

  @classmethod
  def tearDown(cls):
    # tearDown doesn't pass exceptions, so neither can we
    cls._gcontext.__exit__(type=None,value=None,traceback=None)

  def test_init_from_op(self):
    natops = self._g.get_operations()
    tfops = [TFOp.from_root_op(natop) for natop in natops]

    # Check types
    for tfop,natop in zip(tfops,natops):
      assert tfop.optype==natop.type, 'TFOp picked up incorrect operation type: '+str(tfop.optype)+' should be '+str(natop.type)
    assert tfop.id is not None, 'TFOp has an invalid id: '+str(tfop.id)

    # Check id uniqueness
    assert len(tfops)==len(set([_.id for _ in tfops])), 'Non-unique names found in tfops: '+','.join([str(_.id) for _ in tfops])

  def test_params_from_op(self):
    c=0
    for root_op in self._g.get_operations():
      tf_op = TFOp.from_root_op(root_op)
      # Check parameters
      input_tensors = [v for v in tf_op.parameter_values if isinstance(v,T)]
      root_tensors = root_op.inputs
      assert len(input_tensors)==len(root_tensors), 'Mismatch in number of input tensors.'
      for dnnamo_t,root_t in zip(input_tensors, root_tensors):
        try:
          assert dnnamo_t.s==root_t.name, 'Tensor name mismatch'
        except Exception as e:
          # Some op inputs are suffixed with their output tensor slot.
          # This is meaningless, since the names refer to the same thing.
          assert dnnamo_t.s==root_t.name[:-2], 'Tensor name mismatch'
        c+=1
    print 'Checked '+str(c)+' operations successfully.'

  def test_init_from_def(self):
    natops = self._g.get_operations()
    tfops = [TFOp.from_root_def(natop.node_def) for natop in natops]

    # Check types
    for tfop,natop in zip(tfops,natops):
      assert tfop.optype==natop.type, 'TFOp picked up incorrect operation type: '+str(tfop.optype)+' should be '+str(natop.type)
    assert tfop.id is not None, 'TFOp has an invalid id: '+str(tfop.id)

    # Check id uniqueness
    assert len(tfops)==len(set([_.id for _ in tfops])), 'Non-unique names found in tfops: '+','.join([str(_.id) for _ in tfops])


  def test_params_from_def(self):
    c=0
    for root_op in self._g.get_operations():
      tf_op = TFOp.from_root_def(root_op.node_def)
      # Check parameters
      input_tensors = [v for v in tf_op.parameter_values if isinstance(v,T)]
      root_tensors = root_op.inputs
      assert len(input_tensors)==len(root_tensors), 'Mismatch in number of input tensors.'
      for dnnamo_t,root_t in zip(input_tensors, root_tensors):
        try:
          assert dnnamo_t.s==root_t.name, 'Tensor name mismatch'
        except Exception as e:
          # Some op inputs are suffixed with their output tensor slot.
          # This is meaningless, since the names refer to the same thing.
          assert dnnamo_t.s==root_t.name[:-2], 'Tensor name mismatch'
        c+=1
    print 'Checked '+str(c)+' operations successfully.'


  def test_ops_from_different_sources(self):
    for root_op in self._g.get_operations():
      from_op = TFOp.from_root_op(root_op)
      from_def = TFOp.from_root_def(root_op.node_def)
      for p_op,p_def in zip(from_op.parameter_names, from_def.parameter_names):
        assert p_op==p_def, 'Input parameter name mismatch.'
      for p_op,p_def in zip(from_op.parameter_values, from_def.parameter_values):
        assert p_op==p_def, 'Input parameter value mismatch.'


class TestHarder(TestTFOp):
  # Do the same tests as above, except with SimpleNNet instead.
  @classmethod
  def setUp(cls):
    cls._net = SimpleNNet()
    cls._g = cls._net.g
    cls._gcontext = cls._g.as_default()
    cls._gcontext.__enter__()
