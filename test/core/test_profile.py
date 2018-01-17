import unittest

from dnnamo.core.profile import Profile

class TestProfile(unittest.TestCase):
  def test_instantiate(self):
    _ = Profile()

  def test_basics(self):
    p = Profile()

    p['opt_1'] = [1,2,3]
    assert 1 in p['opt_1']
    assert 2 in p['opt_1']
    assert 3 in p['opt_1']

    p['opt_2'] = [3,4,5]
    assert 3 in p['opt_2']
    assert 4 in p['opt_2']
    assert 5 in p['opt_2']
    assert p.consistent

    p.add('opt_3',5)
    p.add('opt_3',6)
    p.add('opt_3',7)
    assert 5 in p['opt_3']
    assert 6 in p['opt_3']
    assert 7 in p['opt_3']
    assert p.consistent

  def test_aggregation(self):
    p = Profile()
    p['a'] = [1,2,3]
    p['b'] = [4,5,6]
    ap = p.aggregate(mode='mean')
    assert ap['a']==2
    assert ap['b']==5
    ap = p.aggregate(mode='first')
    assert ap['a']==1
    assert ap['b']==4
    ap = p.aggregate(mode='last')
    assert ap['a']==3
    assert ap['b']==6
