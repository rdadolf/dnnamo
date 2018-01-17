from abc import ABCMeta, abstractmethod

class _TagMeta(ABCMeta):
  def __str__(self): return str('Datatag:'+str(self.__name__))
class TagName(object):
  __metaclass__=_TagMeta
  @abstractmethod
  def _invalid(self): pass # Do not instantiate Datatag types.
# This let's us designate all the tag classes as "special", which makes it
# possible to identify them in a __dict__ or dir() lookup. It would probably
# have been possible to use Enum's, but NONE and ALL would've been weird. So
# maybe this implementation is a bit more complicated, but the interface that
# everyone else sees is fairly clean.

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
  class graph(TagName): pass
  class absgraph(TagName): pass
  class weights(TagName): pass
  # Dynamic data
  class rungraph(TagName): pass
  class timing(TagName): pass
  class ivalues(TagName): pass # FIXME: Name on this one?

  @classmethod
  def get_all_tags(cls):
    return [v for _,v in cls.__dict__.items() if isinstance(v,_TagMeta)]


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

  def __getitem__(self, tag):
    return self._cache[tag]

  def __setitem__(self, tag, value):
    if value is None:
      raise ValueError, 'Attempted to insert None into data manager tag "'+str(tag)+'". This probably means the corresponding data collector failed.'
    self._cache[tag] = value
