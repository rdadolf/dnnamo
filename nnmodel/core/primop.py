from abc import ABCMeta, abstractmethod

# This is just the interface definition.
class Primop(object):
  __meta__ = ABCMeta

  def __init__(self):
    self.device = None
    self.id = type(self)._unique_id()

  @property
  @abstractmethod
  def args(self): pass

  id_counter = 0

  @classmethod
  def _unique_id(cls):
    c = Primop.id_counter
    Primop.id_counter += 1
    primop_id = str(cls.optype)+'_'+str(c)
    return primop_id

  # @property and @classmethod don't play nicely together.
  # This is the workaround.
  class classproperty(object):
    def __init__(self, func):
      self.func = classmethod(func)
    def __get__(self, *args):
      return self.func.__get__(*args)()

  @classproperty
  def optype(cls):
    # This is a little bit ugly, but it streamlines defining primops.
    s = cls.__name__
    assert s[0:7]=='Primop_', 'Invalid primop name: "'+s+'"'
    assert len(s)>7, 'Invalid primop name: "'+s+'"'
    return s[7:]

  def __str__(self):
    return '<Primop_'+str(self.optype)+':'+str(self.id)+'>'

# This is a primop for undefined operations.
# It's valid in dependence graphs, but will obviously give unrealistic values if
# it is used for modeling. Still, it's useful to let us purposefully under-cover
# the source framework's operation space or intentionally ignore certain cheap
# operations.
class Primop_undef(Primop): pass

##### Basic Linear Algebra Primitives #####

class Primop_mmmul(Primop):
  '''matrix-matrix multiplication, out = AB'''
  def __init__(self, dim_A, dim_B):
    super(Primop_mmmul,self).__init__()
    self.dim_A = dim_A
    self.dim_B = dim_B

  @property
  def args(self):
    return (self.dim_A, self.dim_B)

class Primop_mvmul(Primop):
  '''Matrix-vector multiplication: out = Ab'''
  def __init__(self, dim_A, dim_b):
    super(Primop_mvmul,self).__init__()
    self.dim_A = dim_A
    self.dim_b = dim_b

  @property
  def args(self):
    return (self.dim_A, self.dim_b)

class Primop_vvadd(Primop):
  '''Vector-vector addition: out = a+b'''
  def __init__(self, dim_a, dim_b):
    super(Primop_vvadd,self).__init__()
    self.dim_a = dim_a
    self.dim_b = dim_b

  @property
  def args(self):
    return (self.dim_a, self.dim_b)

##### Neural Network Primitives #####

class Primop_conv(Primop):
  '''Convolution over a matrix M of a filter F'''
  def __init__(self, dim_M, dim_F):
    super(Primop_conv,self).__init__()
    self.dim_M = dim_M
    self.dim_F = dim_F

  @property
  def args(self):
    return (self.dim_M, self.dim_F)
