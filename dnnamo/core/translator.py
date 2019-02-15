from abc import ABCMeta, abstractmethod
from collections import namedtuple
from .primop import Primop_undef, Primop_zero

class Match(object):
  __metaclass__ = ABCMeta
  @abstractmethod
  def match(self, graph, op): 'Returns true if the rule should emit a primop.'

class Emit(object):
  __metaclass__ = ABCMeta
  @abstractmethod
  def emit(self, graph, op): 'Returns a Primop instance from the native op.'

Rule = namedtuple('Rule', ['priority','match_obj','emit_obj'])

class Rules(object):
  rules = []

  @classmethod
  def add(cls, priority, match_obj, emit_obj):
    cls.rules.append( Rule(priority, match_obj, emit_obj) )

################################################################################
# Generic match and emit objects

class MatchAny(Match):
  def match(self, graph, op): return True

class MatchExactType(Match):
  def __init__(self, t): self.t=t
  def match(self, graph, op): return op.type==self.t

class EmitUndef(Emit):
  def emit(self, graph, op): return Primop_undef(root=op)

class EmitZero(Emit):
  def emit(self, graph, op): return Primop_zero(root=op)

################################################################################

class Translator(object):
  '''A graph pattern-matching and rewrite engine.'''
  __metaclass__ = ABCMeta

  def emit_primop(self, RulesClass, graph, op):
    RulesClass.rules.sort(key=lambda r:r.priority)
    for rule in RulesClass.rules:
      if rule.match_obj.match(graph, op):
        return rule.emit_obj.emit(graph, op)
    raise TypeError, 'Missing translation rule for native operation '+str(op)

  @abstractmethod
  def translate(self, graph):
    '''Return a primitive DnnamoGraph object from a native DnnamoGraph object.'''

  #@abstractmethod
  #def map_native_op(self, native_op_id): pass

  #@abstractmethod
  #def map_primop(self, primop_id): pass
