
class DnnamoID(object):
  def __init__(self, s):
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
    c = DnnamoID._id_counter
    DnnamoID._id_counter += 1
    return cls(str(prefix)+'_'+str(c))

class T(DnnamoID):
  '''Unique identifier for Dnnamo Tensor objects'''
  def __str__(self):
    return 'T:'+self.s

class OP(DnnamoID):
  '''Unique identifier for Dnnamo Operation objects'''
  def __str__(self):
    return 'OP:'+self.s
