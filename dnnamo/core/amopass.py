from abc import ABCMeta, abstractmethod, abstractproperty 

class AMOPass(object):
  __metaclass__ = ABCMeta
  '''This is the base class for all analysis, modeling, and optimization methods.'''

  @abstractmethod
  def run(self, frame):
    '''Run the pass. This is where all the heavy lifting happens.'''
  

class AnalysisPass(AMOPass):
  __metaclass__ = ABCMeta

  # analyses are not allowed to invalidate data

class TransformPass(AMOPass):
  __metaclass__ = ABCMeta

  @abstractproperty
  def invalidation_tags(self): 'Data sources this transform invalidates.'

