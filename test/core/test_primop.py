import unittest
import nnmodel

from nnmodel.core.primop import Primop

class TestPrimop(unittest.TestCase):
  def test_primop_names(self):
    # Make sure all nnmodel primops are named Primop_*
    primop_types = [t for t in Primop.__subclasses__() if t.__module__ is nnmodel.core.primop]
    for primop_type in primop_types:
      opname = primop_type.opname()
      assert isinstance(opname,str), 'Invalid primop opname: '+str(opname)
      assert len(opname)>0, 'Invalid primop opname: '+str(opname)
