import unittest
import pytest

from dnnamo.core.bimap import Bimap

class TestBimap(unittest.TestCase):
  def test_instantiation(self):
    b = Bimap()

@pytest.mark.parametrize('b',[
    Bimap([('a',1), ('b',2), ('c',3)]),
    Bimap({'a':1, 'b':2, 'c':3}),
    Bimap(a=1, b=2, c=3),
  ])
class TestBimapOps(object):
  def test_construction(self,b):
      # Elements
      assert b.l['a'] == 1
      assert b.l['b'] == 2
      assert b.l['c'] == 3
      assert b.r[1] == 'a'
      assert b.r[2] == 'b'
      assert b.r[3] == 'c'
      # Whole views
      b.l == {'a':1, 'b':2, 'c':3}
      b.r == {1:'a', 2:'b', 3:'c'}

  def test_mutation(self,b):
    b.l[1] = 'd'
    assert b.l[1] == 'd'
    assert b.r['d'] == 1

  def test_inclusion(self,b):
    # View inclusion
    assert 'a' in b.l
    assert 1 in b.r
    # Bimap inclusion
    assert 'a' in b
    assert 1 in b
