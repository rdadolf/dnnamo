from nnmodel.core.primop import *
from nnmodel.core.dgraph import DGraph
from nnmodel.core.translator import Translator


_RULES = []
def translation_rule(func):
  global _RULES
  _RULES.append(func)
  return func

def translation_rule_matching(optype):
  def translation_rule(func):
    global _RULES
    def f(op):
      if not (op.type==optype):
        return None
      return func(op)
    _RULES.append(f)
    return f
  return translation_rule

################################################################################
# Tensorflow translation rules
# NOTE: ORDER MATTERS! Rules will be pattern-matched according to the order in
#       which they are declared in this file.
# NOTE: ALWAYS wrap a translation rule with the @translation_rule decorator.

# ---TEMPLATE---
#@translation_rule
#def convert_MatMul(op):
#  # PATTERN
#  if not matching-condition-using-op-goes-here
#    return None
#  # RULE
#  return Primop_OPNAME( op.appropriate-argument-conversions )

@translation_rule_matching('MatMul')
def convert_MatMul(op):
  dim_A = op.inputs[0].get_shape().as_list()
  dim_B = op.inputs[1].get_shape().as_list()
  if op.get_attr('transpose_a'):
    dim_A = dim_A[::-1]
  if op.get_attr('transpose_b'):
    dim_B = dim_B[::-1]
  return Primop_mmmul(dim_A,dim_B)

# FIXME: Rule disabled. time_mvmul benchmark must handle >2-dimensional tensors
#@translation_rule_matching('BiasAdd')
def convert_BiasAdd(op):
  dim_A = op.inputs[0].get_shape().as_list()
  dim_b = op.inputs[1].get_shape().as_list()[0]
  return Primop_mvmul(dim_A,dim_b)

#FIXME: Rule disabled. time_conv() benchmark must handle >2-dimensional tensors
#@translation_rule_matching('Conv2D')
def convert_Conv2D(op):
  dim_F = op.inputs[0].get_shape().as_list()
  dim_M = op.inputs[1].get_shape().as_list()
  #op.get_attr('strides') # FIXME
  #op.get_attr('padding') # FIXME
  return Primop_conv(dim_M, dim_F)

# TODO: Top-90% native ops
# CIFAR10:
#   Conv2DBackpropFilter
#   Conv2D
#   Conv2DBackpropInput
#   MaxPoolGrad
#   LRNGrad
#   MatMul
#   LRN
#   BiasAdd


# NOTE: This MUST be the last rule.
@translation_rule
def default(op):
  # NO PATTERN
  # RULE
  return Primop_undef()
# End of Tensorflow translation rules
################################################################################

class TFTranslator(Translator):
  def __init__(self):
    self._dgraph = None
    self._map = {}
    self._rmap = {}

  def translate(self, tf_model):
    if self._dgraph is not None:
      return self._dgraph
    self._dgraph = DGraph()

    tf_graph = tf_model.model()
    # Add all ops as nodes in the dependence graph.
    for op in tf_graph.get_operations():
      for rule in _RULES:
        primop = rule(op)
        if primop is not None:
          break
      else:
        raise TypeError('No translation rule found for native operation: '+str(op))
      self._dgraph.add_primop( primop )
      self._map[op.name] = primop.id
      self._rmap[primop.id] = op.name

    # Now add edges between all ops.
    # FIXME
    # TF doesn't have explicit op-op edges. It uses tensors as intermediaries.
    # Tensors can have multiple consumers, so they act as hyperedges.
    # To avoid adding duplicate edges, we only add an edge from the source node.
    for src_tf_op in tf_graph.get_operations():
      for tensor in src_tf_op.outputs:
        for dst_tf_op in tensor.consumers():
          src_primop = self._dgraph[self._map[src_tf_op.name]]
          dst_primop = self._dgraph[self._map[dst_tf_op.name]]
          self._dgraph.add_dep( src_primop, dst_primop )

    return self._dgraph

  def map_native_op(self, native_op_id):
    return self._map[native_op_id]

  def map_primop(self, primop_id):
    return self._rmap[primop_id]


