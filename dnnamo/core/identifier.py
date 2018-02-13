class DnnamoID(object):

  def __init__(self, s):
    if str(s)!=s:
      raise TypeError, 'Identifiers must be created from a string, not a '+str(type(s))
    self._s = s

  @property
  def s(self):
    return self._s
  _id_counter = 0

  @classmethod
  def unique(cls,prefix=None):
    '''Unique ID factory.'''
    if prefix is None:
      prefix='ID'
    # Identifier counter increments *globally*, across all ID types.
    c = DnnamoID._id_counter
    DnnamoID._id_counter += 1
    return cls(str(prefix)+'_'+str(c))

  def __eq__(self, other):
    if type(self)!=type(other):
      return False
    return self.s==other.s

  def __ne__(self, other):
    # Python's != calls this directly, so we need to override it
    return not(self==other)

  def __hash__(self):
    # Allow use of DnnamoID's as keys, but don't hash different ID's the same way
    return hash( (self.__class__.__name__, self.s) )

  def __repr__(self):
    return str(self.__class__.__name__)+"('"+self.s+"')"

  def __str__(self):
    return self.s

# Shortcut
ID = DnnamoID

### Specific ID types

#class T(DnnamoID):
#  '''Unique identifier for Dnnamo Tensor objects'''

#class OP(DnnamoID):
#  '''Unique identifier for Dnnamo Operation objects'''

#class DependenceID(DnnamoID):
#  '''Unique identifier for Dnnamo Dependence objects'''

#class ProxyID(DnnamoID):
#  '''Unique identifier for Dnnamo Proxy objects'''
