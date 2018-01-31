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

  def test_equality(self):
    o1 = OP('a')
    o2 = OP('a')
    o3 = OP('b')
    t1 = T('a')
    t2 = T('a')
    t3 = T('b')

    # Check equality between OP id's
    assert o1==o2
    assert o1 is not o2
    assert o1!=o3
    assert o1 is not o3

    # Check equality between T id's
    assert t1==t2
    assert t1 is not t2
    assert t1!=t3
    assert t1 is not t3

    # Check equality between T's and OP's
    assert t1.s==o1.s
    assert t1!=o1
    assert t2.s==o2.s
    assert t2!=o2
    assert t3.s==o3.s
    assert t3!=o3
    assert t1.s==o2.s
    assert t1!=o2


  def test_hashability(self):
    o1 = OP('a')
    o2 = OP('a')
    o3 = OP('b')
    t1 = T('a')
    t2 = T('a')
    t3 = T('b')

    # Check hashability between OP id's
    d = dict()
    d[o1] = 1
    d[o2] = 2
    d[o3] = 3
    assert d[o1]==d[o2]
    assert d[o1]==2
    assert d[o2]==2
    assert d[o1]!=d[o3]
    assert d[o2]!=d[o3]
    assert d[o3]==3

    # Check hashability between OP id's
    d = dict()
    d[t1] = 1
    d[t2] = 2
    d[t3] = 3
    assert d[t1]==d[t2]
    assert d[t1]==2
    assert d[t2]==2
    assert d[t1]!=d[t3]
    assert d[t2]!=d[t3]
    assert d[t3]==3

    # Check hashability between T's and OP's
    d = dict()
    d[o1] = 1
    d[o2] = 2
    d[o3] = 3
    d[t1] = 4
    d[t2] = 5
    d[t3] = 6

    assert len(d)==4
    assert d[o1]==2
    assert d[o2]==2
    assert d[o3]==3
    assert d[t1]==5
    assert d[t2]==5
    assert d[t3]==6

    assert d[o1]!=d[t1]
    assert d[o2]!=d[t2]
    assert d[o3]!=d[t3]
