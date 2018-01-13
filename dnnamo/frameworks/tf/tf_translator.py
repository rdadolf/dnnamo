from dnnamo.core.primop import *
from dnnamo.core.absgraph import AbstractGraph
from dnnamo.core.translator import Rules, Match, Emit, Translator, MatchAny, EmitUndef, EmitZero

################################################################################
# Match components

class MatchExactType(Match):
  def __init__(self, t):
    self.t=t
  def match(self, op):
    return op.type==self.t

################################################################################
# Emit components

class EmitUnaryKronecker(Emit):
  def emit(self, op):
    return Primop_kronecker({'dim': op.inputs[0].get_shape().as_list()}, source_op=op)

class EmitBinaryKronecker(Emit):
  def emit(self, op):
    dim_a = op.inputs[0].get_shape().as_list()
    dim_b = op.inputs[1].get_shape().as_list()
    if len(dim_a)<1:
      dim_a = dim_b
    elif len(dim_b)<1:
      dim_b = dim_a
    # if both are <1, both are already [ ]
    # if both are >1, then TF disallows input tensors with different dimensions
    return Primop_kronecker({'dim': dim_a}, source_op=op)

class EmitDot2D(Emit):
  def emit(self, op):
    dim_a = op.inputs[0].get_shape().as_list()
    dim_b = op.inputs[1].get_shape().as_list()
    # FIXME: Check if these transpose ops incur a performance cost
    #   If they're just indexing tricks, then the below works.
    if op.get_attr('transpose_a'):
      dim_a = dim_a[::-1]
    if op.get_attr('transpose_b'):
      dim_b = dim_b[::-1]
    return Primop_dot({'dim_a':dim_a, 'dim_b':dim_b, 'dim_reduce':1}, source_op=op)

################################################################################
# Translation rules

class TFRules(Rules): pass

TFRules.add(99, MatchAny(), EmitUndef())

TFRules.add(50, MatchExactType('NoOp'), EmitZero())

TFRules.add(50, MatchExactType('Const'), EmitZero()) #?
TFRules.add(50, MatchExactType('Placeholder'), EmitZero()) #?

TFRules.add(50, MatchExactType('Add'), EmitBinaryKronecker())
TFRules.add(50, MatchExactType('Sub'), EmitBinaryKronecker())
TFRules.add(50, MatchExactType('Mul'), EmitBinaryKronecker())
TFRules.add(50, MatchExactType('Div'), EmitBinaryKronecker())
TFRules.add(50, MatchExactType('FloorDiv'), EmitBinaryKronecker())
TFRules.add(50, MatchExactType('RealDiv'), EmitBinaryKronecker())
TFRules.add(50, MatchExactType('TruncateMod'), EmitBinaryKronecker())
TFRules.add(50, MatchExactType('FloorMod'), EmitBinaryKronecker())

TFRules.add(50, MatchExactType('MatMul'), EmitDot2D())

#Rule(50, MatchExactType('Conv2D'), EmitConv2d())

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

# End of Tensorflow translation rules
################################################################################

class TFTranslator(Translator):
  def __init__(self):
    self._map = {}
    self._rmap = {}

  def translate(self, model):
    # Caching is handled at the framework level, so if this translate method is
    # called, it means we definitely want to rebuild the abstract graph.
    absgraph = AbstractGraph()

    tf_graph = model.get_graph()
    # Add all ops as nodes in the dependence graph.
    for op in tf_graph.get_operations():
      primop = self.emit_primop(TFRules, op)
      absgraph.add_primop( primop )
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
          src_primop = absgraph[self._map[src_tf_op.name]]
          dst_primop = absgraph[self._map[dst_tf_op.name]]
          absgraph.add_dep( src_primop, dst_primop )

    return absgraph

  def map_native_op(self, native_op_id):
    return self._map[native_op_id]

  def map_primop(self, primop_id):
    return self._rmap[primop_id]


