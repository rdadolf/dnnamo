import re

from ...core.bimap import Bimap
from ...core.graph import DnnamoGraph
from ...core.dataflow import DnnamoOp, DnnamoProxy, DnnamoTensor, DnnamoDependence
from ...core.identifier import ID
#from .tf_op import TFOp
#from .tf_tensor import TFTensor

class TFGraph(DnnamoGraph):
  def __init__(self):
    super(TFGraph, self).__init__()
    self._vertex_name = Bimap() # tf name -> dnnamo ID
    self._tensor_name = Bimap() # tf name -> dnnamo ID
    self._dependence_name = Bimap() # tf name -> dnnamo ID
    # NOTE: some operations produce outputs which are used as both dataflow
    #   AND control-flow edges. Dnnamo graphs are hypergraphs, but they don't
    #   have a provision for split-type hyperedges (where some endpoints are
    #   dependences and others are tensors). However, Dnnamo graphs are also
    #   multi-graphs, so having both types of hyperedges between vertices is
    #   just fine.

  # User-visible constructor methods

  @classmethod
  def from_graph(cls, root_graph):
    '''Creates a new TFGraph from a TensorFlow graph object.'''
    g = cls()
    g.augment_from_graph(root_graph)
    return g

  @classmethod
  def from_rmd(self, rmd):
    '''Creates a new TFGraph from a TensorFlow RunMetadata protobuf.'''
    g = TFGraph()
    g.augment_from_rmd(rmd)
    return g

  def augment_from_graph(self, root_graph):
    self._augment_graph(root_graph=root_graph)

  def augment_from_rmd(self, root_rmd):
    self._augment_graph(root_rmd=root_rmd)

  # These functions do the actual work for constructing graphs.

  def _convert_tf_attribute(self, attr):
    'Converts a TF attr_value protobuf object into a reasonable python object.'
    # FIXME: NYI. Passing through isn't the worst thing, but it should be fixed
    return attr

  def _is_dependence(self, edge_string):
    return edge_string.startswith('^')

  def _stripped_edgename(self, edge_string):
    # NOTE: One of the reasons we have this function is to allow conversion from
    # vertex names to edge names. The RMD node_stats have no way of telling
    # whether its outputs are used as tensors or dependence edges. If we store
    # edges as stripped names, we don't need to track it.
    if edge_string.startswith('^'):
      return edge_string[1:]
    return edge_string

  def _idemp_strip_slot(self, edge_string):
    m = re.match('(.*):[0-9]+$',edge_string)
    if m:
      return m.group(1)
    return edge_string

  def _idemp_add_slot(self, edge_string):
    if re.match('.*:[0-9]+$', edge_string):
      return edge_string
    return edge_string+':0'

  def _tensorshape_to_dnnamo_shape(self, tensorshape_proto):
    if tensorshape_proto.unknown_rank:
      return None
    else:
      return [int(d.size) for d in tensorshape_proto.dim]

  def _augment_graph(self, root_graph=None, root_rmd=None):
    assert (root_graph is not None) or (root_rmd is not None), 'Must supply at least one root data structure to augment_graph'

    if root_graph:
      nodedefs = [node for node in root_graph.as_graph_def().node]
    else:
      nodedefs = [node for part in root_rmd.partition_graphs for node in part.node]

    ### (1) Add stub vertices, populate vertex name table
    # NOTE: parameter lists are empty
    for node in nodedefs:
      i = ID.unique(node.name)
      self._vertex_name.l[node.name] = i
      t = node.op
      if t.startswith('_'):
        v = DnnamoProxy(id=i, type=t, root=node)
        self.add_proxy(v)
      else:
        v = DnnamoOp(id=i, type=t, parameters=[], root=node) # empty params
        self.add_op(v)
    # FIXME: Add node_stat vertices here (just _SOURCE?)

    ### (2) Add stub edges, populate edge name tables
    # NOTE: edge shapes are empty
    # We'll build up a list first, then add them all
    edgelist = {} # edge name -> [ src, [(dst,dep?), ...] ]
    for node in nodedefs:
      for edgename in node.input:
        # Distill edge ID from name
        dependence = self._is_dependence(edgename)
        stripped_edgename = self._stripped_edgename(edgename)
        # Check if we've seen this edge before
        if stripped_edgename not in edgelist:
          # New edge, find its source and create it
          # First, convert edge name to source vertex ID
          src = self._vertex_name.l[self._idemp_strip_slot(stripped_edgename)]
          edgelist[stripped_edgename] = [src, []]
        # Add destination vertex ID (dsts list might not be complete, though, so
        #   don't call self.add_edge() yet)
        dst = self._vertex_name.l[node.name]
        edgelist[stripped_edgename][1].append( (dst,dependence) )

    # Now add all the edges
    for stripped_edgename,(src, dst_pairs) in edgelist.items():
      tensor_dsts = [dst for dst,dep in dst_pairs if not dep]
      dep_dsts = [dst for dst,dep in dst_pairs if dep]
      # Create a DnnamoTensor and DnnamoDependence hyper edge if they both exist
      if len(tensor_dsts)>0:
        id = ID.unique(stripped_edgename)
        self._tensor_name.l[stripped_edgename] = id
        # If we're dealing with a graph, look up the root tensor
        if root_graph:
          # TF doesn't use ^'s in tensor names and expects explicit slot numbers
          tensorname = self._idemp_add_slot(stripped_edgename)
          root = root_graph.get_tensor_by_name(tensorname)
        else:
          root = stripped_edgename # For RMD's, we just use the original string
        self.add_tensor(DnnamoTensor(id, None, [src], tensor_dsts, root)) # empty shape
      if len(dep_dsts)>0:
        id = ID.unique(stripped_edgename)
        self._dependence_name.l[stripped_edgename] = id
        # Control-flow edges never have a root object, we just use the string
        root = stripped_edgename
        self.add_dependence(DnnamoDependence(id, [src], dep_dsts, root))


    ### (3) Fill op parameters
    #   Accurate edge IDs can now be looked up from edge name table

    for op in self.ops: # Proxy vertices don't have parameters)
      nodedef = op.root
      # Parameter names are integer strings ('0', '1', ...) for TF inputs
      #   followed by TF attribute names, in alphabetical order
      inputs = []
      for i,input in enumerate(nodedef.input):
        name = self._stripped_edgename(input)
        if self._is_dependence(input):
          inputs.append( (str(i), self._dependence_name.l[name]) )
        else:
          inputs.append( (str(i), self._tensor_name.l[name]) )
      #input_names  = [k for k,_ in inputs]
      #input_values = [v for _,v in inputs]

      attr_dict = dict(nodedef.attr)
      sorted_attr_pairs = sorted(attr_dict.items(), key=lambda (k,v): k)
      attrs = [(str(k),self._convert_tf_attribute(v)) for k,v in sorted_attr_pairs]
      #attr_names  = [k for k,_ in attrs]
      #attr_values = [v for _,v in attrs]

      op.set_parameters( inputs + attrs )

    ### (4) Fill edge shape information

    if root_graph:
      # Tensors from a tf.Graph have shape information in their graph
      for tensor in self.tensors:
        tfshape = tensor.root.get_shape().as_proto()
        tensor.set_shape( self._tensorshape_to_dnnamo_shape(tfshape) )

    elif root_rmd:
      # Tensors from a RMD have shape information in step_stats
      for nodestat in [_ for dev in root_rmd.step_stats.dev_stats for _ in dev.node_stats]:
        for i,output in enumerate(nodestat.output):
          # Convert vertex name -> edge name -> edge ID
          edgename = self._stripped_edgename(nodestat.node_name)
          if i>0:
            edgename += ':'+str(i)
          if edgename in self._tensor_name.l:
            edgeid = self._tensor_name.l[edgename]
          else: # XXX
            # There exist some edges which are *only* used in control flow
            # contexts but which have real, allocated shapes in the profile.
            # It is unclear whether the data is actually passed between ops
            # in these cases (and also unclear why the hell TF did this).
            #
            # In Dnnamo, we are currently treating these edges as pure
            # control flow dependences, not data flow.
            continue # do not add shape to DnnamoDependence edges

          # A paranoid invariant...
          assert i==output.slot, 'Slot mismatch in protobuf: '+str(nodestat)
          # Grab shape
          tfshape = output.tensor_description.shape
          shape = self._tensorshape_to_dnnamo_shape(tfshape)
          self.tensor(edgeid).set_shape(shape)

    # END _augment_graph



  #def augment_from_graph(self, root_graph):
  #  '''Adds elements from a TensorFlow graph.'''
  #  for root_op in root_graph.get_operations():

  #    tf_op = TFOp.from_root_op(root_op)
  #    self.add_op(tf_op)
  #    # All TensorFlow tensors are produced by a single operation, so
  #    # we don't need to worry about adding this tensor twice. (They can
  #    # be consumed twice, but that's not relevant here.)
  #    for root_tensor in root_op.values():
  #      tf_t = TFTensor.from_root_tensor(root_tensor)
  #      self.add_tensor(tf_t)
  #  return self



  #def augment_from_rmd(self, rmd):
  #  '''Adds elements from a TensorFlow RunMetadata protobuf.

  #  RunMetadata protobufs are the result of profiling the execution of a
  #  TensorFlow graph. This method makes a best-effort reconstruction of
  #  the graph that was run. The data TensorFlow provides can be incomplete.
  #  '''
  #  # At a high level, graph nodes (ops) are held in the 'partition_graph'
  #  # field, and memory/timing information is held in the 'step_stats' field.
  #  # Unfortunately, these structures are inconsistent at best. Fields often
  #  # do not exist, and many values must be inferred.

  #  ### (1) collect operations from the partition graphs.

  #  # Segregate the pseudo-ops (any optype that begins with '_') into a
  #  # separate list for later.
  #  _transmission_ops = {} # _Send, _Recv
  #  _source_ops = {} # _Arg_* (XXX: Are these only placeholders or just anything that is replaced by a feed_dict argument?)
  #  _sink_ops = {} # _Result

  #  for part in rmd.partition_graphs:
  #    for node in part.node:
  #      if node.op.startswith('_'):
  #        self.add_proxy( TFProxy(id=node.name, type=node.op, root=node) )
  #      #  pass # FIXME: Segregate
  #        print 'SEGREGATING',node
  #      else:
  #        tfop = TFOp.from_root_def(node)
  #        self.add_op(tfop)

  #  ### (2) collect tensor connectivity information
  #  # Find consumers (dsts)
  #  tensor_dsts = {} # tensor_name -> [dst, ...]
  #  for op in self.ops:
  #    for p in op.parameter_values:
  #      if isinstance(p,T):
  #        if p not in tensor_dsts:
  #          tensor_dsts[p] = []
  #        tensor_dsts[p].append(op.id)
  #  # Associate any tensor with a source op
  #  # NOTE: since we're only creating tensor connections based on what was
  #  #   actually used, there could be ops which don't produce anything
  #  tensor_srcs = {} # tensor_name -> [src]
  #  for t in tensor_dsts:
  #    # Create an OP ID from the T ID---they should be basically the same
  #    src_id = OP(TFOp._opname_from_tensorname(t.s))
  #    if src_id not in self:
  #      raise ValueError('Tensor '+str(t)+' produced by unknown op (best guess is '+str(src_id)+')')
  #    tensor_srcs[t] = src_id

  #  ### (3) collect tensor shape information from step stats
  #  tensor_shapes = {}
  #  tensor_roots = {}
  #  for dev in rmd.step_stats.dev_stats:
  #    for node in dev.node_stats:
  #      output_tensor_prefix = node.node_name
  #      for (i,out_t) in enumerate(node.output):
  #        # Add a slot tag for every output
  #        output_tensor_slot = output_tensor_prefix+':'+str(i)
  #        # Canonicalize tensor name (removes slot on :0)
  #        output_tensor_name = TFTensor._fix_tensorname(output_tensor_slot)
  #        # Assume every output is a tensor
  #        tensor = out_t.tensor_description
  #        t_id = T(output_tensor_name)
  #        tensor_shapes[t_id] = [int(d.size) for d in tensor.shape.dim]
  #        tensor_roots[t_id] = out_t # tensor description protobuf

  #  ### (4) unify all tensor information
  #  # Check if everyone has the same tensors.
  #  src_keys = set(tensor_srcs.keys())
  #  dst_keys = set(tensor_dsts.keys())
  #  shp_keys = set(tensor_shapes.keys())
  #  # tensor_roots have same entries as shape, so don't need to check
  #  assert len(src_keys ^ dst_keys)==0
  #  assert len(src_keys ^ shp_keys)==0

  #  for t in src_keys:
  #    tensor = TFTensor(t, tensor_shapes[t], tensor_srcs[t], tensor_dsts[t], root=None)
  #    self.add_tensor(tensor)

  #  return self


