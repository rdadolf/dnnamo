from .identifier import T, OP
from .op import DnnamoOp
from .tensor import DnnamoTensor

class DnnamoGraph(object):
  '''A generic directed, attributed graph class for computational graphs.

  Nodes of a graph are operations; edges are tensors.'''

  def __init__(self):
    self._ops = {} # { OP => DnnamoOp }
    self._tensors = {} # { T => DnnamoTensor }
    self._adj = {} # { OP => { OP=>[T, ...], OP=>[T, ...], ... }, ... }

  ### Built-ins

  def __get__(self, ident):
    '''Retrieve the object corresponding to an identifier.

    For OP identifiers, this returns a DnnamoOp.
    For T identifiers, this returns a DnnamoTensor.'''
    if isinstance(ident, OP):
      return self._ops[ident]
    elif isinstance(ident, T):
      return self._tensors[ident]
    else:
      raise TypeError('Identifier must be either a tensor (T) or operation (OP) identifier: '+str(ident))

  def __contains(self, ident):
    if isinstance(ident, OP):
      return ident in self._ops
    elif isinstance(ident, T):
      return ident in self._tensors
    else:
      raise TypeError('Identifier must be either a tensor (T) or operation (OP) identifier: '+str(ident))

  def __str__(self):
    return '<Graph OP('+str(len(self.ops))+') T('+str(len(self.tensors))+')>'

  ### Iterators
  # Generators are great, but they can also cause problems, for instance
  # len() doesn't work on them. Since we don't have supermassive graphs,
  # we just return lists. 

  @property
  def ops(self):
    '''Return all DnnamoOp's in the graph.'''
    return self._ops.values()

  @property
  def tensors(self):
    '''Return all DnnamoTensor's in the graph.'''
    return self._tensors.values()

  def ops_outgoing_from(self, op_id):
    return self._adj[op_id].keys()

  def ops_incoming_to(self, op_id):
    return [_ for _ in self.ops if op_id in self._adj[_.id]]

  # NOTE:
  # Hyper-multi-graphs can have the same edge coming out of a node twice and
  # have the same the edge coming from two different nodes. At the same time.
  # 
  # OP1()->T, OP2()->T, OP3(t)->[], OP4(t)->[]
  # OP1         OP3(T)
  #    \       /
  #     >--T--<
  #    /       \
  # OP2         OP4(T)
  #
  # OP3 and OP4 both consume tensor T, but T can be produced by either OP1
  # or OP2 (perhaps there is a conditional controlling the source).
  #
  # These functions should only list tensors once, since they only exist once.

  def tensors_outgoing_from(self, op_id):
    return [_ for _ in self._tensors if op_id in _.dsts]

  def tensors_incoming_to(self, op_id):
    return [_ for _ in self._tensors if op_id in _.srcs]

  ### Mutators

  def add_op(self, op):
    if op.id in self._ops:
      raise KeyError('Duplicate operation ID found in graph: '+str(op.id))
    self._ops[op.id] = op
    self._adj[op.id] = {}

  def add_tensor(self, tensor):
    if tensor.id in self._tensors:
      raise KeyError('Duplicate tensor ID found in graph: '+str(tensor.id))
    self._tensors[tensor.id] = tensor
    for src in tensor.srcs:
      for dst in tensor.dsts:
        # Appending is safe because we guarantee that the ID is unique
        if dst not in self._adj[src]:
          self._adj[src][dst] = []
        self._adj[src][dst].append(tensor.id)


  # FIXME: More methods here?
