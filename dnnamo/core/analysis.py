from abc import ABCMeta, abstractmethod, abstractproperty 

class AnalysisResult(object):
  __metaclass__ = ABCMeta

  '''The result of any analysis operation.'''
  pass

class Analysis(object):
  __metaclass__ = ABCMeta

  def __init__(self):
    pass

  @abstractproperty
  def invalidation_tags(self):
    '''Data tags which, when invalidated, require this analysis to be re-run.

    These should be class constants in AnalysisInvalidationTag.'''
