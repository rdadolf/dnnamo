import unittest

from dnnamo.core.registry import Registry

class ExampleRegistry(Registry): pass

class TestRegistry(unittest.TestCase):
  def test_register(self):
    ExampleRegistry._deregister_all()

    with self.assertRaises(KeyError):
      ExampleRegistry.lookup('a')
    with self.assertRaises(KeyError):
      ExampleRegistry.rlookup(1)

    ExampleRegistry.register('a', 1)

    assert 1==ExampleRegistry.lookup('a')
    assert 'a'==ExampleRegistry.rlookup(1)

  def test_register(self):
    ExampleRegistry._deregister_all()

    # Forwards
    with self.assertRaises(KeyError):
      ExampleRegistry.lookup('b')
    ExampleRegistry.register('b', 2)
    assert 2==ExampleRegistry.lookup('b')
    ExampleRegistry._deregister('b')
    with self.assertRaises(KeyError):
      ExampleRegistry.lookup('b')

    # Backwards
    ExampleRegistry.register('b', 2)
    assert 2==ExampleRegistry.lookup('b')
    ExampleRegistry._rderegister(2)
    with self.assertRaises(KeyError):
      ExampleRegistry.lookup('b')

    # All
    ExampleRegistry.register('b', 2)
    ExampleRegistry.register('c', 3)
    assert 2==ExampleRegistry.lookup('b')
    assert 3==ExampleRegistry.lookup('c')
    ExampleRegistry._deregister_all()
    with self.assertRaises(KeyError):
      ExampleRegistry.lookup('b')
    with self.assertRaises(KeyError):
      ExampleRegistry.lookup('c')

  def test_iterate(self):
    ExampleRegistry._deregister_all()

    ExampleRegistry.register('a',1)
    ExampleRegistry.register('b',2)
    ExampleRegistry.register('c',3)
    assert len(ExampleRegistry)==3
    for key in ExampleRegistry:
      assert type(key)==str
      assert key in ExampleRegistry
      assert ExampleRegistry.lookup(key) is not None
