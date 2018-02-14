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

  ### Name translation support
  def get_vertex_id_from_tf_name(self, tf_name):
    '''Returns the DnnamoVertex constructed from a TensorFlow name.

    Dnnamo uses globally-unique identifiers for its graph elements, whereas
    TensorFlow reuses names for operations, tensors, and other objects. This
    function translates the TensorFlow name for a graph node into a Dnnamo ID.'''
    return self._vertex_name.l[tf_name]

  def get_tensor_id_from_tf_name(self, tf_name):
    '''Returns the DnnamoTensor constructed from a TensorFlow name.

    Dnnamo uses globally-unique identifiers for its graph elements, whereas
    TensorFlow reuses names for operations, tensors, and other objects. This
    function translates the TensorFlow name for a tensor into a DnnamoID.
    Note: Tensorflow sometimes augments tensor names with syntactic sugar,
    including a slot suffix and a control-flow prefix. This method requires
    the correct slot suffix (or '' for slot 0), and it will strip any
    control-flow prefix off the name. Note that if a tensor is used *only*
    for control-flow, this method with return an error.'''
    return self._tensor_name.l[tf_name]

  ### User-visible constructor methods

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

  ### These functions do the actual work for constructing graphs.

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

