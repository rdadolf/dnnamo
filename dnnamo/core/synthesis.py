from abc import ABCMeta, abstractmethod
from .model import DnnamoModel

class SyntheticModel(DnnamoModel):
  __metaclass__ = ABCMeta

  def __init__(self, exemplar):
    self._exemplar = exemplar

  @property
  def exemplar(self):
    return self._exemplar
