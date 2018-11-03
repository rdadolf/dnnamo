from .bimap import Bimap

class IterableType(type): # metaclass to enable __iter__ on class objects
  def __iter__(self):
    return self._class_iterator()
  def __len__(self):
    return self._class_len()

class Registry(object):
  '''Registries are common in Dnnamo. It would be nice if they used the same interface.'''
  __metaclass__ = IterableType

  _registry = None

  @classmethod
  def _idempotent_allocate_registry(cls):
    # Allocating this as part of the part class, such as using a class attribute,
    # will cause subclasses to inherit (and SHARE!) the parent class's storage.
    # This is not the behavior we want. So instead, we use a None sentinel for an
    # un-initialized registry, then idempotently create the registry on the first
    # use of it.
    # NOTE: We call this on *EVERY* function, even the ones that don't make much
    #   sense to do so. Sure, it's overkill, but it also makes the error messages
    #   a bit more meaningful ("cls does not have a _regstiry attr" vs. KeyError
    #   in, e.g., deregistration.
    if cls._registry is None:
      cls._registry = Bimap()

  @classmethod
  def _class_len(cls):
    return len(cls._registry)

  @classmethod
  def _class_iterator(cls):
    cls._idempotent_allocate_registry()
    for _ in cls._registry.l:
      yield _

  @classmethod
  def register(cls, lhs, rhs):
    '''Registers the hashable key 'lhs' with the hashable key 'rhs'.

    This is a bijective map, and it is held as part of the class, not an instance.'''
    cls._idempotent_allocate_registry()
    cls._registry.l[lhs] = rhs

  @classmethod
  def _deregister(cls, lhs):
    '''Deregister a key pair using the 'lhs' key. This is mostly a testing function.'''
    cls._idempotent_allocate_registry()
    del cls._registry.l[lhs]

  @classmethod
  def _rderegister(cls, rhs):
    '''Deregister a key pair using the 'rhs' key. This is mostly a testing function.'''
    cls._idempotent_allocate_registry()
    del cls._registry.r[rhs]

  @classmethod
  def _deregister_all(cls):
    '''Deregister all key pairs. This is mostly a testing function.'''
    cls._registry = None
    cls._idempotent_allocate_registry()

  @classmethod
  def lookup(cls, lhs):
    '''Lookup a value using the 'lhs' key originally used to register it.'''
    cls._idempotent_allocate_registry()
    return cls._registry.l[lhs]

  @classmethod
  def rlookup(cls, rhs):
    '''Reverse-lookup a value using the 'rhs' key originally used to register it.'''
    cls._idempotent_allocate_registry()
    return cls._registry.r[rhs]

  @classmethod
  def items(cls):
    cls._idempotent_allocate_registry()
    return cls._registry.l.items()

  def keys(kcs):
    cls._idempotent_allocate_registry()
    return cls._registry.l.keys()

  def values(kcs):
    cls._idempotent_allocate_registry()
    return cls._registry.r.keys()
