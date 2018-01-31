import tensorflow as tf

from ...core.op import DnnamoOp
from ...core.identifier import T, OP

class TFOp(DnnamoOp):

  @classmethod
  def from_root_op(cls, root_graph, root_op):
    '''Create a TFOp from a TensorFlow Operation object.

    Arguments
      root_graph: a TensorFlow Graph object. This is required for tensor information.
      root_op: a TensorFlow Operation object. This must be part of the graph argument.'''

    # Input tensors are strings
    input_tensor_values = [T(_.name) for _ in root_op.inputs]
    # TensorFlow is useless when it comes to trying to match input dataflow 
    # dependencies with operation signatures. OpDef is supposed to do this,
    # but it doesn't always exist, and when it does, it doesn't always match
    # the input tensors in a sane way (c.f. the horrific nonsense that is the
    # ArgDef protobuf inside OpDefs). So we just ignore it altogether.
    # XXX: If becomes a problem, we've got a world of pain waiting for us.
    input_tensor_names = ['T'+str(_) for _ in xrange(0,len(input_tensor_values))]

    input_attributes = [(k,v) for k,v in root_op.node_def.attr.items()]
    # Input attributes are held in arbitrary order, so we just pick one.
    # TensorFlow cannot tell the difference, so long as we pass them as
    # keyword arguments during reconstruction or synthesis.
    if len(input_attributes)>0:
      input_attr_names,input_attr_values = [list(_) for _ in zip(*input_attributes)]
    else:
      input_attr_names,input_attr_values = [],[]

    names = input_tensor_names+input_attr_names
    values = input_tensor_values+input_attr_values
    params = [(k,v) for k,v in zip(names, values)]

    # TF op names are already unique
    return cls(id=root_op.name, optype=root_op.type, parameters=params, root=root_op)

  def to_root_op(self, tfgraph, root_graph):
    pass # FIXME
    # node_def
    # g
    # inputs
    # output_types
    # return tf.Opeation(...)

  def __str__(self):
    return '<TFOp_'+str(self.optype)+':'+str(self.id)+'>'
