import tensorflow as tf
from tensorflow.python.framework.ops import get_stats_for_node_def
import numpy as np

from nnmodel.core.stats import NativeStats
# FIXME: NYI from nnmodel.core.trace import average_traces


def copy_tf_graph(graph):
  graph_proto = graph.as_graph_def()
  copy = tf.Graph()
  with copy.as_default():
    tf.import_graph_def(graph_proto, name='')
  return copy


class TFNativeStats(NativeStats):
  def __init__(self, model, traces):
    self._model = model
    self._traces = traces
    # Parse collected data and generate useful data structures
    self._build_meantrace() # _meantrace
    self._build_tensormaps() # _tensors, _src_tensormap, _dst_tensormap
    self._build_static_proxy() # _static_proxy
    self._build_static_stats() # _flops, _params

  def _build_meantrace(self):
    # XXX FIXME XXX: this is not a mean.
    self._meantrace = self._traces[1]

  def _build_tensormaps(self):
    # Supported stats:
    self._tensors = {} # tensor_name -> {'stat':value, ... }
    #   supported tensor stats: 'typesize', 'dims'
    self._src_tensormap = {} # src_op -> [ out_tensor_name, ... ]
    self._dst_tensormap = {} # dst_op -> [ in_tensor_name, ... ]

    # First, populate the src/dst maps via static analysis
    graph = self._model.model()
    for op in graph.get_operations():
      self._src_tensormap[op.name] = [t.name for t in op.outputs]
      for t in op.outputs:
        if t.name not in self._tensors:
          self._tensors[t.name] = {}
        self._tensors[t.name]['typesize'] = t.dtype.size
      self._dst_tensormap[op.name] = [t.name for t in op.inputs]
      for t in op.inputs:
        if t.name not in self._tensors:
          self._tensors[t.name] = {}
        self._tensors[t.name]['typesize'] = t.dtype.size
    # Now match runtime shape information to the statically-constructed maps
    for tracepoint in self._meantrace:
      src_op = tracepoint.name
      static_tensor_names = self._src_tensormap[src_op]
      dynamic_tensor_dims = tracepoint.tensor_dims

      # If only one dynamic output, any extra static outputs are multiple uses
      if len(dynamic_tensor_dims)==1:
        dims = dynamic_tensor_dims[0]
        for t in static_tensor_names:
          self._tensors[t]['dims'] = dims
      elif len(dynamic_tensor_dims)>1:
        # FIXME: Ambiguity in TF traces: see github issue #34
        assert len(static_tensor_names)<=len(dynamic_tensor_dims), 'Ambiguity in TF trace output: cannot assign a dynamic output tensor to a static graph tensor'
        for t,dims in zip(static_tensor_names, dynamic_tensor_dims):
          self._tensors[t]['dims'] = dims
      else:
        # No output tensors were recorded. It's not entirely clear why this is
        # happening, but I believe it's due to either explicit control flow
        # dependences and/or zero-sized output tensors.

        # FIXME: If we ever see a non-zero, un-traced tensor. Change this block.
        print tracepoint, tracepoint.tensor_dims, len(dynamic_tensor_dims)
        static_outputs = graph.get_operation_by_name(src_op).outputs
        shapes = [t.get_shape().as_list() for t in static_outputs]
        print shapes
        #for s in shapes:
        #  assert None not in s, 'Partially-specified shape found in op '+str(src_op)+' '+str(shapes)
        #static_products = [np.prod(s) for s in shapes]
        #assert sum(static_products)==0, 'Non-zero-sized output tensors found for an op that had no dynamic trace tensors.'
        #for t in static_tensor_names:
        #  self._tensors[t]['dims'] = [0]

  def _build_static_proxy(self):
    # NOTE: This gets hairy.
    #   Tensorflow doesn't handle computing static stats on partially-defined
    # tensors. Since we still want to leverage the built-in stat functions
    # (tf.python.framework.ops.get_stats_for_node_def...), we need a compromise:
    #   1. Create a full, op-for-op copy of the graph
    #   2. Run the original graph out to create a dynamic trace of all the ops
    #   3. Extract the dynamic shape information from the trace
    #   4. On the copy, update every partially-defined tensor to have the same
    #      shape as was recorded in the trace.
    #   5. Now we have a fully-specified static graph. Run the built-in stats
    #      functions on this copy and store it. The name-op mapping should be
    #      valid for the original graph as well.

    # 1 Create a copy
    g = self._model.model()
    h = copy_tf_graph(g)
    # 2 Grab a trace (given)
    #for tp in self._meantrace:
    #  print tp
    # 3 Extract shape info (done in _build_tensormaps)
    # 4 Update partially-defined tensors
    # n.b.- we only receive shape information from the outputs of tensors (this
    #   is an artifact of the RunMetadata protobuf), so we only bother updating
    #   op output tensors.
    for t,t_stats in self._tensors.items():
      tensor = h.get_tensor_by_name(t)
      if 'dims' in t_stats:
        # FIXME: is_compatible only exists in TF versions beyond 0.8.0.
        #        Uncomment this check when TF version is bumped in nnmodel.
        #assert tensor.get_shape().is_compatible(t_stats['dims']), 'Dynamically-traced tensor shape information ('+str(t_stats['dims'])+') is incompatible with static graph shape ('+str(tensor.get_shape())+')'
        tensor.set_shape(t_stats['dims'])
      else:
        if not tensor.get_shape().is_fully_defined():
          print 'Warning: partially-specified tensor was not updated with dynamic shape information. Subsequent stats on this tensor and all downstream ops may be incorrect: '+str(tensor.name)
    self._static_proxy = h

  def _build_static_stats(self):
    '''Collect static information from the graph and collect it into tables.'''
    # Supported stats:
    self._flops = {}
    self._params = {}
    self._bytes = {}

    graph = self._static_proxy
    for op in graph.get_operations():
      if not all([t.get_shape().is_fully_defined() for t in op.inputs]):
        self._flops[op.name] = 0
        self._params[op.name] = 0
        self._bytes[op.name] = 0
        print 'Warning: "'+str(op.name)+'" has invalid statistics'
        continue
      flops = get_stats_for_node_def(graph, op.node_def, 'flops').value or 0
      params = get_stats_for_node_def(graph, op.node_def, 'params').value or 0
      bytes = 0
      for t in op.inputs:
        typesize = t.dtype.size
        dims = np.prod(t.get_shape().as_list()) or 0
        bytes += typesize * dims
      self._flops[op.name] = flops
      self._params[op.name] = params
      self._bytes[op.name] = bytes

  def computational_density(self, native_op_name):
    # Get all input tensor stats
    try:
      bytes = self._bytes[native_op_name]
    except KeyError: # Op wasn't in static graph
      bytes = 0

    # Get flop information
    try:
      flops = self._flops[native_op_name]
    except KeyError: # Op wasn't in static graph
      flops = 0

    return float(flops), float(bytes)
