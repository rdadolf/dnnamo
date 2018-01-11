from abc import ABCMeta, abstractmethod, abstractproperty

class _TagMeta(type): pass
class Datatag(object):
  '''Categories of data.

  These categories describe different types of data produced by collectors and
  used by analyses. The AnalysisManager tracks the validity of its cache using
  these tags, and various transforms can cause that cache to be invalidated.
  These tags are also used to run DataCollectors when the data is deemed to be
  needed by an analysis.'''

  # Pseudo-tags
  class NONE(object): pass # not actually tags
  class ALL(object): pass # not actually tags
  # Static data
  class graph(object): __metaclass__=_TagMeta # FIXME: do we even need this?
  class absgraph(object): __metaclass__=_TagMeta
  class weights(object): __metaclass__=_TagMeta
  # Dynamic data
  class timing(object): __metaclass__=_TagMeta
  class ivalues(object): __metaclass__=_TagMeta # FIXME: Name on this one?

  @classmethod
  def get_all_tags(cls):
    return [k for k,v in cls.__dict__.items() if isinstance(v,_TagMeta)]


class DataManager(object):
  registry = {} # class => [Datatag, ...]

  @classmethod
  def register(cls, new_class):
    if new_class not in cls.registry:
      cls.registry[new_class] = new_class().invalidation_tags

  @classmethod
  def _deregister(cls, victim_class):
    # Deregistration is not a normal use case. Mostly just for test.
    del cls.registry[victim_class]

  def __init__(self):
    self._cache = {_:None for _ in Datatag.get_all_tags()} # str => value | None

  def invalidate(self, tag):
    if tag is Datatag.ALL:
      eviction_list = Datatag.get_all_tags()
    elif tag is Datatag.NONE:
      eviction_list = []
    else:
      eviction_list = [tag]

    for victim in eviction_list:
      self._cache[victim] = None

