import tensorflow as tf

from ...core.op import DnnamoOp
from ...core.identifier import T, OP
from .tf_tensor import TFTensor

class TFOp(DnnamoOp):
  @classmethod
  def _canonicalize_tensors(cls,tensors):
    if len(tensors)==0:
      return []
    # Tensor parameter keys are [T0, T1, ...]
    # Why?
    # TensorFlow is useless when it comes to trying to match input dataflow
    # dependencies with operation signatures. OpDef is supposed to do this,
    # but it doesn't always exist, and when it does, it doesn't always match
    # the input tensors in a sane way (c.f. the horrific nonsense that is the
    # ArgDef protobuf inside OpDefs). So we just ignore it altogether.
    # XXX: If becomes a problem, we've got a world of pain waiting for us.
    try:
      s = tensors[0].name
      # We're dealing with a _InputList of tf.Tensor objects
      accessor = lambda t: t.name
    except AttributeError:
      # we're dealing with a NodeDef RepeatedScalarContainer of input strings
      accessor = lambda t: t

    tensornames = [TFTensor._fix_tensorname(accessor(t)) for t in tensors]
    # Remove control edges from input lists
    real_tensors = [T(t) for t in tensornames if not t.startswith('^')]
    return [('T'+str(i), t) for i,t in enumerate(real_tensors)]

  @staticmethod
  def _canonicalize_attrs(attrs):
    # Attribute key-values are in arbitrary order.
    # Why?
    # TensorFlow holds attributes in a map in arbitrary order. Since we do not,
    # we just pick an arbitrary order and stick to it.
    return [(k,v) for k,v in attrs.items()]

  @classmethod
  def _from_input_lists(cls, id, optype, tensors, attrs, root):
    tensors = cls._canonicalize_tensors(tensors)
    attrs = cls._canonicalize_attrs(attrs)
    names = [k for k,_ in tensors] + [k for k,_ in attrs]
    values = [v for _,v in tensors] + [v for _,v in attrs]
    params = [(k,v) for k,v in zip(names, values)]
    return cls(id=id, optype=optype, parameters=params, root=root)

  @classmethod
  def from_root_def(cls, root_def):
    '''Create a TFOp from a TensorFlow NodeDef object.'''
    return cls._from_input_lists(
      id = root_def.name,
      optype = root_def.op,
      tensors = root_def.input,
      attrs = root_def.attr,
      root = root_def
    )

  @classmethod
  def from_root_op(cls, root_op):
    '''Create a TFOp from a TensorFlow Operation object.'''
    return cls._from_input_lists(
      id = root_op.name,
      optype = root_op.type,
      tensors = root_op.inputs,
      attrs = root_op.node_def.attr,
      root = root_op
    )

  def to_root_op(self, tfgraph, root_graph):
    # FIXME: This may or may not by impossible.
    pass
    # Arguments to tf.Operation:
    #   node_def
    #   g
    #   inputs
    #   output_types
    # return tf.Opeation(...)

  def __str__(self):
    return '<TFOp_'+str(self.optype)+':'+str(self.id)+'>'
