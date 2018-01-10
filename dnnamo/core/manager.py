from abc import ABCMeta, abstractmethod, abstractproperty

class GenericManager(object):
  __metaclass__ = ABCMeta

  registry = {} # class => [InvalidationTag, ...]

  @classmethod
  def register(cls, new_class):
    if new_class not in cls.registry:
      cls.registry[new_class] = new_class().invalidation_tags

  @classmethod
  def _deregister(cls, victim_class):
    # Deregistration is not a normal use case. Mostly just for test.
    del cls.registry[victim_class]


class InvalidationTag(object):
  '''Categories of data which are tracked by the analysis manager.

  These categories describe different types of data produced by a data collector
  and used by analyses.  When a transform causes a model to change or a trigger
  asks, these categories determine which analyses are no longer valid. The 
  analyses are then re-executed lazily.'''

  NONE = 0
  ALL = -1
  WEIGHT_VALUES = 1
  GRAPH_STRUCTURE = 2


class AnalysisManager(GenericManager):
  def __init__(self):
    self._cache = {_:None for _ in self.registry.keys()} # str => AnalysisResult | None
    pass

  def invalidate(self, tag_to_invalidate):
    eviction_list = [cls for cls,tags in self.registry.items() \
                         if ((InvalidationTag.ALL==tag_to_invalidate) \
                         or  (InvalidationTag.NONE!=tag_to_invalidate \
                              and tag_to_invalidate in tags) \
                         or  (InvalidationTag.NONE!=tag_to_invalidate \
                              and InvalidationTag.ALL in tags))]
    for victim in eviction_list:
      self._cache[victim] = None

  def run(self, analysis_name, model, trigger):
    if trigger=='always':
      self._cache[analysis_class]


class TransformManager(GenericManager):
  def __init__(self):
    pass

  def run(self, analysis_class, model):
    pass
