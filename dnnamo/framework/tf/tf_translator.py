from dnnamo.core.bimap import Bimap
from dnnamo.core.primop import *
from dnnamo.core.graph import DnnamoGraph
from dnnamo.core.dataflow import DnnamoTensor
from dnnamo.core.translator import Rules, Match, Emit, Translator, \
                                   MatchAny, MatchExactType, \
                                   EmitUndef, EmitZero

################################################################################
# Match components


################################################################################
# Emit components

class EmitUnaryHadamard(Emit):
  def emit(self, op):
    dim = graph.tensor(op.parameter_values[0]).shape
    return Primop_hadamard({'dim': dim}, root=op)
    #return Primop_hadamard({'dim': op.inputs[0].get_shape().as_list()}, root=op)

class EmitBinaryHadamard(Emit):
  def emit(self, graph, op):
    dim_a = graph.tensor(op.parameter_values[0]).shape
    dim_b = graph.tensor(op.parameter_values[1]).shape
    #dim_a = op.inputs[0].get_shape().as_list()
    #dim_b = op.inputs[1].get_shape().as_list()
    if len(dim_a)<1:
      dim_a = dim_b
    elif len(dim_b)<1:
      dim_b = dim_a
    # if both are <1, both are already [ ]
    # if both are >1, then TF disallows input tensors with different dimensions
    return Primop_hadamard({'dim': dim_a}, root=op)

class EmitDot2D(Emit):
  def emit(self, graph, op):
    dim_a = graph.tensor(op.parameter_values[0]).shape
    dim_b = graph.tensor(op.parameter_values[1]).shape
    #dim_a = op.inputs[0].get_shape().as_list()
    #dim_b = op.inputs[1].get_shape().as_list()
    # FIXME: Check if these transpose ops incur a performance cost
    #   If they're just indexing tricks, then the below works.
    if op.parameters['transpose_a'].b:
      dim_a = dim_a[::-1]
    if op.parameters['transpose_b'].b:
      dim_b = dim_b[::-1]
    return Primop_dot({'dim_a':dim_a, 'dim_b':dim_b, 'dim_reduce':1}, root=op)

################################################################################
# Translation rules

class TFRules(Rules): pass

TFRules.add(99, MatchAny(), EmitUndef())

TFRules.add(50, MatchExactType('NoOp'), EmitZero())

TFRules.add(50, MatchExactType('Const'), EmitZero()) #?
TFRules.add(50, MatchExactType('Placeholder'), EmitZero()) #?

TFRules.add(50, MatchExactType('Add'), EmitBinaryHadamard())
TFRules.add(50, MatchExactType('Sub'), EmitBinaryHadamard())
TFRules.add(50, MatchExactType('Mul'), EmitBinaryHadamard())
TFRules.add(50, MatchExactType('Div'), EmitBinaryHadamard())
TFRules.add(50, MatchExactType('FloorDiv'), EmitBinaryHadamard())
TFRules.add(50, MatchExactType('RealDiv'), EmitBinaryHadamard())
TFRules.add(50, MatchExactType('TruncateMod'), EmitBinaryHadamard())
TFRules.add(50, MatchExactType('FloorMod'), EmitBinaryHadamard())

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
    self._map = Bimap() # Left: native, Right: primitive

  def translate(self, graph):
    # Caching is handled at the framework level, so if this translate method is
    # called, it means we definitely want to rebuild the abstract graph.
    primgraph = DnnamoGraph()

    # Add all ops to the graph.
    for op in graph.ops:
      primop = self.emit_primop(TFRules, graph, op)
      primgraph.add_op(primop)
      self._map.l[op.id] = primop.id

    # Add all proxy vertices as zero nodes.
    for proxy in graph.proxies:
      primop = EmitZero().emit(graph, proxy)
      primgraph.add_op(primop) # FIXME? Is this the right type?
      self._map.l[proxy.id] = primop.id

    # Add all tensors as edges in the dependence graph.
    for t in graph.tensors:
      prim_srcs = [self._map.l[_] for _ in t.srcs]
      prim_dsts = [self._map.l[_] for _ in t.dsts]
      primt = DnnamoTensor(t.id.s, t.shape, prim_srcs, prim_dsts, root=t)
      primgraph.add_tensor(primt)

    # Omit control-flow edges.

    return primgraph

    # Add all ops as nodes in the dependence graph.
    #for op in graph.get_operations():
    #  primop = self.emit_primop(TFRules, op)
    #  absgraph.add_primop( primop )
    #  self._map[op.name] = primop.id
    #  self._rmap[primop.id] = op.name

    # Now add edges between all ops.
    # FIXME
    # TF doesn't have explicit op-op edges. It uses tensors as intermediaries.
    # Tensors can have multiple consumers, so they act as hyperedges.
    # To avoid adding duplicate edges, we only add an edge from the source node.
    #for src_tf_op in graph.get_operations():
    #  for tensor in src_tf_op.outputs:
    #    for dst_tf_op in tensor.consumers():
    #      src_primop = absgraph[self._map[src_tf_op.name]]
    #      dst_primop = absgraph[self._map[dst_tf_op.name]]
    #      absgraph.add_dep( src_primop, dst_primop )

    #return absgraph

