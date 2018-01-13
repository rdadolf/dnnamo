from abc import ABCMeta, abstractmethod
from collections import namedtuple
from .primop import Primop_undef, Primop_zero

class Match(object):
  __metaclass__ = ABCMeta
  @abstractmethod 
  def match(self, op): 'Returns true if the rule should emit a primop.'
  
class Emit(object):
  __metaclass__ = ABCMeta
  @abstractmethod 
  def emit(self, op): 'Returns a Primop instance from the native op.'

Rule = namedtuple('Rule', ['priority','match_obj','emit_obj'])

class Rules(object):
  rules = []
  
  @classmethod
  def add(cls, priority, match_obj, emit_obj):
    cls.rules.append( Rule(priority, match_obj, emit_obj) )

################################################################################
# Generic match and emit objects

class MatchAny(Match):
  def match(self, op): return True

class EmitUndef(Emit):
  def emit(self, op): return Primop_undef(source_op=op)

class EmitZero(Emit):
  def emit(self, op): return Primop_zero(source_op=op)

################################################################################

class Translator(object):
  '''A graph pattern-matching and rewrite engine.'''
  __metaclass__ = ABCMeta

  def emit_primop(self, RulesClass, op):
    RulesClass.rules.sort(key=lambda r:r.priority)
    for rule in RulesClass.rules:
      if rule.match_obj.match(op):
        return rule.emit_obj.emit(op)
    raise TypeError, 'Missing translation rule for native operation '+str(op)

  @abstractmethod
  def translate(self, model):
    '''Return an AbstractGraph object from a Dnnamo model.'''

  @abstractmethod
  def map_native_op(self, native_op_id): pass

  @abstractmethod
  def map_primop(self, primop_id): pass
