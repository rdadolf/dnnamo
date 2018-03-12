import unittest

from dnnamo.core.primop import Primop, PrimopTypes

class TestPrimop(unittest.TestCase):
  def test_primop_names(self):
    # Make sure all dnnamo primops are named Primop_*
    for type,Op in PrimopTypes().items():
      assert issubclass(Op, Primop), 'Found a Primop that is not a Primop: ('+str(Op)+')'
      assert isinstance(type,str), 'Invalid primop opname: '+str(type)

  def test_null_instantiation(self):
    for type,Op in PrimopTypes().items():
      p = Op()
      assert p.type==type, 'Internal type doesnt match global entry.'
      assert len(p.parameters)==len(p.parameter_names), 'Parameter names and values dont match up'

  def test_parameter_instantiation(self):
    for _,Op in PrimopTypes().items():
      pnames = Op().parameter_names # instantiation is necessary for properties
      p = Op([1 for _ in pnames])

      for k,v in p.parameters.items():
        assert v==1, 'Error in creating parameters: '+str(k)+'='+str(v)+' (should be 1)'
