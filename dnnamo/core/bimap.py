class Unimap(dict):
  def _set_inverse(self, inv):
    self._inv = inv

  def _uni_setitem(self, k, v):
    super(Unimap,self).__setitem__(k,v)

  def __setitem__(self, k, v):
    super(Unimap,self).__setitem__(k,v)
    # Must be set only one way, otherwise goes into infinite recursion
    self._inv._uni_setitem(v,k)


class Bimap(object):
  '''A bidirectional, bijective map class.

  This class stores a mapping from one set of keys to another (there are no
  dedicated values). We call one set the "left-hand" keys and the other the
  "right-hand" keys. Each key maps to exactly one key with opposite
  handedness/chirality.

  The map cannot be accessed or iterated through directly, since there is no
  canonical set of keys. Instead, the user must choose a view first. This
  is done with the Bimap.l and Bimap.r properties. These views can then be
  used as normal dictionaries, with the sole caveat that mutation affects
  both (since they are just views of the same structure).'''

  def __init__(self, *args, **kwargs):
    '''Bimap supports all constructor mechanisms that a normal dict does.

    The arguments provided will be mapped left-to-right. For instance:
      b = Bimap( {1:2, 3:4} )
      b = Bimap( [(1,2), (3,4)] )
    will both have the following views:
      b.l == {1:2, 3:4}
      b.r == {2:1, 4:3}'''

    self._l = Unimap()
    self._r = Unimap()
    self._l._set_inverse(self._r)
    self._r._set_inverse(self._l)
    d = dict(*args, **kwargs) # Trick to get exact dict constructor behavior
    for k,v in d.items():
      self.l[k] = v

  def __contains__(self, k):
    return (k in self._l) or (k in self._r)

  @property
  def l(self):
    '''Views the bimap as a map from left-hand keys to right-hand keys.

    Supports all operations dicts do, including assignment. Assignment
    to a left-hand view will be reflected in the right-hand view. '''
    return self._l

  @property
  def r(self):
    '''Views the bimap as a map from right-hand keys to left-hand keys.

    Supports all operations dicts do, including assignment. Assignment
    to a right-hand view will be reflected in the left-hand view. '''
    return self._r
