from .dataflow import DnnamoDFO, DnnamoOp, DnnamoTensor, DnnamoVertex, DnnamoEdge

class DnnamoGraph(object):
  '''A generic directed, attributed graph class for computational graphs.'''

  _VERTEX_TYPES = set(['op','proxy'])
  _EDGE_TYPES = set(['tensor','dependence'])

  def __init__(self):
    # Graph resources
    self._res = {} # ID => object
    self._restype = {} # ID => 'op' | 'proxy' | 'tensor' | 'dependence'
    # Graph structure
    self._vout = {} # v => [e, ...] (edges outgoing from vertex)
    self._vin = {} # v => [e, ...] (edges incoming to vertex)
    self._eout = {} # e => [v, ...] (vertices outgoing from edge)
    self._ein = {} # e => [v, ...] (vertices incoming from edge)

    # NOTE:
    # Hyper-multi-graphs can have the same edge exiting twice out of the same
    # vertex. This is weird, but it actually can happen if an op takes the same
    # tensor as two different arguments.
    # So the incidence lists *CAN* have duplicate entries:
    #   self._vin[V] = [e1,e1,e2]  # Valid. Op taking one tensor for two args.
    #   self._eout[E] = [v1,v1,v2] # Valid. As above, but from the tensor's PoV.
    # The dual (_vout/_ein) are weird but allowed in Dnnamo. (I don't think they
    # show up.


  ### Checks
  def _assert_invariants(self):
    # Global
    assert len(self._vout)==len(self._vin), \
      'Vertex adjacency lists must exist, even if they are empty'
    assert len(self._eout)==len(self._ein), \
      'Edge incidence lists must exist, even if they are empty'
    assert len(self._vout)+len(self._eout)==len(self._res), \
      'Adjacency and incidence lists should sum to total resources'
    assert len(self._res)==len(self._restype), \
      'Resources and resource types should have the same cardinality'
    # Local
    for v,elist in self._vin.items():
      for e in elist:
        assert v in self._eout[e], \
          'An edge incoming from a vertex must have that vertex outgoing from it'
    for v,elist in self._vout.items():
      for e in elist:
        assert v in self._ein[t], \
          'An edge outgoing to a vertex must have that vertex incoming to it'
    # FIXME
          #'A vertex incident from an edge must have that edge incoming to it'

  ### Built-ins

  def __getitem__(self, ident):
    '''Retrieve the object corresponding to an identifier.'''
    if ident not in self._res: # Better error message
      raise KeyError('Identifier '+str(ident)+' does not exist in graph')
    return self._res[ident]

  def __contains__(self, ident):
    if ident in self._res:
      return True

  def __str__(self):
    return '<Graph V('+str(len(self._vout))+') E('+str(len(self._eout))+')>'

  ### Resource selectors
  def is_vertex(self, id):
    self[id] # Better nonexistence error message
    return self._restype[id] in self._VERTEX_TYPES
  def is_edge(self, id):
    self[id] # Better nonexistence error message
    return self._restype[id] in self._EDGE_TYPES
  def _istype(self, id, t):
    self[id] # Better nonexistence error message
    return self._restype[id]==t

  @property
  def vertices(self):
    return [res for id,res in self._res.items() if self.is_vertex(id)]
  @property
  def edges(self):
    return [res for id,res in self._res.items() if self.is_edge(id)]
  @property
  def ops(self):
    '''Return all DnnamoOp's in the graph.'''
    return [res for id,res in self._res.items() if self._istype(id,'op')]
  @property
  def tensors(self):
    '''Return all DnnamoTensor's in the graph.'''
    return [res for id,res in self._res.items() if self._istype(id,'tensor')]
  @property
  def dependences(self):
    '''Return all non-tensor dependences in the graph.'''
    return [res for id,res in self._res.items() if self._istype(id,'dependence')]
  @property
  def proxies(self):
    '''Return all non-op proxy vertices in the graph.'''
    return [res for id,res in self._res.items() if self._istype(id,'proxy')]

  def vertices_from(self, id):
    '''All destination vertices for a given edge.'''
    return [v for v in self._eout[id]]
  def vertices_to(self, id):
    '''All source vertices for a given edge.'''
    return [v for v in self._ein[id]]
  def edges_from(self, id):
    '''All edges coming from a given vertex.'''
    return [e for e in self._vout[id]]
  def edges_to(self, id):
    '''All edges leading to a given vertex.'''
    return [e for e in self._vin[id]]
  def ops_from(self, id):
    '''All source ops for a given edge.'''
    return [v for v in self._eout[id] if self._istype(v,'op')]
  def ops_to(self, id):
    '''All destination ops for a given edge.'''
    return [v for v in self._ein[id] if self._istype(v,'op')]
  def tensors_from(self, id):
    '''All tensors produced by a given vertex.'''
    return [e for e in self._vout[id] if self._istype(e,'tensor')]
  def tensors_to(self, id):
    '''All tensors feeding into a given vertex.'''
    return [e for e in self._vin[id] if self._istype(e,'tensor')]
  def dependences_from(self, id):
    '''All dependence edges coming from a given vertex.'''
    return [e for e in self._vout[id] if self._istype(e,'dependence')]
  def dependences_to(self, id):
    '''All dependence edges leading to a given vertex.'''
    return [e for e in self._vin[id] if self._istype(e,'dependence')]

  ### Typesafe Lookups

  def tensor(self, tensor_id):
    '''Look up a tensor by identifier. Returns a DnnamoTensor object.

    A graph can also be indexed directly, but this method typechecks ID first.'''
    t = self[tensor_id]
    if self._restype[tensor_id]!='tensor':
      raise KeyError('Identifier is not a Tensor ID')
    if not isinstance(t,DnnamoTensor):
      raise TypeError('Identifier was stored as a tensor, but holds a non-tensor object: '+str(t))
    return self[tensor_id]

  def op(self, op_id):
    '''Look up an operation by identifier. Returns a DnnamoOp object.

    A graph can also be indexed directly, but this method typechecks ID first.'''
    op = self[op_id]
    if self._restype[op_id]!='op':
      raise KeyError('Identifier is not an operation ID')
    if not isinstance(op,DnnamoOp):
      raise TypeError('Identifier was stored as an op, but holds a non-op object: '+str(op))
    return op

  ### Mutators
  def _add_resource(self, r, typestring):
    if not isinstance(r, DnnamoDFO):
      raise TypeError(str(r)+' is not a valid dataflow object')
    if r.id in self._res:
      raise KeyError('Duplicate ID found in graph: '+str(r.id))
    self._res[r.id] = r
    self._restype[r.id] = typestring

  def add_vertex(self, v, typestring):
    if not isinstance(v, DnnamoVertex):
      raise TypeError(str(v)+' is not a valid dataflow vertex')
    self._add_resource(v, typestring)
    self._vout[v.id] = []
    self._vin[v.id] = []

  def add_edge(self, e, typestring):
    if not isinstance(e, DnnamoEdge):
      raise TypeError(str(e)+' is not a valid dataflow edge')
    for s in e.srcs:
      self[s] # Better nonexistence error message
      if not self.is_vertex(s):
        raise TypeError('Edge points to non-vertex ID: '+str(s))
    for d in e.dsts:
      self[d] # Better nonexistence error message
      if not self.is_vertex(d):
        raise TypeError('Edge comes from non-vertex ID: '+str(d))
    self._add_resource(e, typestring)
    self._eout[e.id] = list(e.srcs)
    self._ein[e.id] = list(e.dsts)
    for s in e.srcs:
      self._vout[s].append(e.id)
    for d in e.dsts:
      self._vin[d].append(e.id)

  def add_op(self, op):
    self.add_vertex(op, 'op')

  def add_proxy(self, proxy):
    self.add_vertex(proxy, 'proxy')

  def add_tensor(self, tensor):
    self.add_edge(tensor, 'tensor')

  def add_dependence(self, dep):
    self.add_edge(dep, 'dependence')
