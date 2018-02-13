import unittest

from dnnamo.core.identifier import ID

class TestIdentifier(unittest.TestCase):
  # Some non-complicated tests just to make sure that identifiers are taking
  # unique values when they should and exact values otherwise.

  def test_exactness(self):
    a = ID('same')
    b = ID('same')
    assert a.s == b.s, 'ID initializers corrupted strings given'

    c = ID('diff1')
    d = ID('diff2')
    assert c != d, 'ID initializers corrupted strings given'

  def test_uniqueness(self):

    a = ID.unique('test')
    b = ID.unique('test')
    c = ID.unique('asdf')
    assert a.s != b.s
    assert a.s not in [_.s for _ in [b,c]], 'ID value should be unique.'
    assert b.s not in [_.s for _ in [a,c]], 'ID value should be unique.'
    assert c.s not in [_.s for _ in [a,b]], 'ID value should be unique.'

  def test_equality(self):
    a = ID('a')
    b = ID('a')
    c = ID('b')
    assert a==b
    assert a is not b
    assert a!=c
    assert a is not c

  def test_hashability(self):
    a = ID('a')
    b = ID('a')
    c = ID('b')
    d = dict()
    d[a] = 1
    d[b] = 2
    d[c] = 3
    assert d[a]==d[b]
    assert d[a]==2
    assert d[b]==2
    assert d[a]!=d[c]
    assert d[b]!=d[c]
    assert d[c]==3
