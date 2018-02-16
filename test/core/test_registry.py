import unittest

from dnnamo.core.registry import Registry

class ExampleRegistry(Registry): pass

class TestRegistry(unittest.TestCase):
  def test_register(self):
    with self.assertRaises(KeyError):
      ExampleRegistry.lookup('a')
    with self.assertRaises(KeyError):
      ExampleRegistry.rlookup(1)

    ExampleRegistry.register('a', 1)

    assert 1==ExampleRegistry.lookup('a')
    assert 'a'==ExampleRegistry.rlookup(1)

  def test_register(self):
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
    

  def test_iterate(self):
    pass # FIXME
