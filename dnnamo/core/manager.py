from abc import ABCMeta, abstractmethod, abstractproperty

class GenericManager(object):
  __metaclass__ = ABCMeta

  registry = {} # class => [InvalidationTag, ...]

  @classmethod
  def _register(cls, new_class):
    if new_class not in cls.registry:
      cls.registry[new_class] = new_class().invalidation_tags

  @classmethod
  def _deregister(cls, victim_class):
    del cls.registry[victim_class]


class InvalidationTag(object):
  '''Categories of data which are tracked by the analysis manager.

  These categories describe different types of data produced by a data collector
  and used by analyses.  When a transform causes a model to change or a trigger
  asks, these categories determine which analyses are no longer valid. The 
  analyses are then re-executed lazily.'''

  NONE = 0
  WEIGHT_VALUES = 1
  GRAPH_STRUCTURE = 2


class AnalysisManager(GenericManager):
  registered_analyses = {} # str => Analysis class
  invalidation_map = {} # str => [InvalidationTag, ...]

  def __init__(self):
    self._cache = {} # str => AnalysisResult | None
    pass

  def invalidate(self, invalidation_tag):
    pass

  def run(self, analysis_name, model, trigger):
    if trigger=='always':
      self._cache[analysis_class]
    pass

#  @classmethod
#  def _register(cls, analysis_name, analysis_class):
#    print 'REGISTERING', analysis_name, analysis_class
#    print '  (REGISTRY:',cls.registered_analyses,')'
#    if analysis_name in cls.registered_analyses:
#      print 'NAME CONFLICT:',cls.registered_analyses[analysis_name],analysis_class
#      if cls.registered_analyses[analysis_name] is analysis_class:
#        # Allow duplicate registration
#        pass
#      else:
#        # Do not allow name collisions or re-definitions of analyses
#        # This is either an import problem or a analysis name selection problem.
#        # Either way, it is a problem and should be fixed.
#        raise NameError, 'Analysis "'+str(analysis_name)+'" is already registered as '+str(cls.registered_analyses[analysis_name])+ ' (attempted to register '+str(analysis_class)+')'

#    cls.registered_analyses[analysis_name] = analysis_class

#    _ = analysis_class() # construct a throw-away object to get at properties
#    tags = _.invalidation_tags
#    if len(tags)<1:
#      raise ValueError, 'Every Analysis must have a set of invalidation tags, even if it is only [InvalidationTag.NONE]. The Analysis '+str(analysis_class)+' declared its tags as: '+str(_)

#    cls.invalidation_map[analysis_name] = [_ for _ in tags if _ is not InvalidationTag.NONE]

#  @classmethod
#  def _deregister(cls, analysis_name):
#    if analysis_name in cls.registered_analyses:
#      # Using only the first test catches cases when a class is partially-
#      # registered (which implies that it's corrupted).
#      del cls.registered_analyses[analysis_name]
#      del cls.invalidation_map[analysis_name]


class TransformManager(GenericManager):
  def __init__(self):
    pass

  def run(self, analysis_class, model):
    pass
