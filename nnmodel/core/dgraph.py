class DGraph(object):
  '''A Dependence Graph, describing the relation between computational primitives (primops).'''

  def __init__(self):
    self.primops = {} # list(primop, primop, ...)
    self._adj = {} # { primop_id: set(primop_id, primop_id, ...), ... }
    #self._gensym = self._Gensym()

  @property
  def devices(self):
    return set([primop.device for primop in self.primops.values()])

  def __len__(self):
    return len(self.primops)

  def __contains__(self, primop_id):
    return primop_id in primops

  def __iter__(self):
    for p in self.primops.values():
      yield p

  def __getitem__(self, primop_id):
    return self.get_primop_by_id(primop_id)

  def add_primop(self, p):
    '''Adds a primop to the graph and returns it.'''
    assert p.id not in self.primops, 'Primop already added to DGraph.'
    self.primops[p.id] = p
    return p

  def add_dep(self, p0, p1):
    '''Adds a directed dataflow dependence edge between p0 and p1.'''
    if p0 not in self._adj:
      self._adj[p0] = set([p1])
    else:
      self._adj[p0].add(p1)

  def get_primop_by_id(self, primop_id):
    return self.primops[primop_id]


  def dep(self, p0):
    '''Return a list of the dependencies of p0.'''
    return self._adj.get(p0,set())

  ### EXPR ###################################################
  def eval(self, variable): # FIXME: constraint
    if variable=='time':
      return self.time()
    raise KeyError('No such variable: '+str(variable))

  def time(self): # FIXME: constraint
    # EXPR
    return sum([primop.time() for primop in self.primops])
