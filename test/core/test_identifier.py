import unittest

from dnnamo.core.identifier import T, OP

class TestIdentifier(unittest.TestCase):
  # Some non-complicated tests just to make sure that identifiers are taking
  # unique values when they should and exact values otherwise.

  def test_exactness(self):
    op_same = OP('same')
    t_same = T('same')
    assert op_same.s == t_same.s, 'ID initializers corrupted strings given'

    op_diff = OP('diff1')
    t_diff = OP('diff2')
    assert op_diff != t_diff, 'ID initializers corrupted strings given'

  def test_uniqueness(self):
    op_diff = OP.unique('diff')
    t_diff = T.unique('diff')
    assert op_diff.s != t_diff.s, 'Identifiers not unique across tensor/op types.'

    o1 = OP.unique('test')
    o2 = OP.unique('test')
    t1 = T.unique('test')
    t2 = T.unique('test')
    assert o1.s not in [_.s for _ in [o2,t1,t2]], 'ID value should be unique.'
    assert o2.s not in [_.s for _ in [o1,t1,t2]], 'ID value should be unique.'
    assert t1.s not in [_.s for _ in [o1,o2,t2]], 'ID value should be unique.'
    assert t2.s not in [_.s for _ in [o1,o2,t1]], 'ID value should be unique.'
