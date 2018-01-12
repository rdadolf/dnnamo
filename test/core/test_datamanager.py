import unittest

from dnnamo.core.amopass import TransformPass
from dnnamo.core.datamanager import Datatag, DataManager

class TestDataManager(unittest.TestCase):
  def test_instantiation(self):
    _ = DataManager

  def test_registration(self):
    class GoodPass(TransformPass):
      def run(self, frame): pass
      @property
      def invalidation_tags(self): return [Datatag.NONE]
    class GoodPass2(TransformPass):
      def run(self, frame): pass
      @property
      def invalidation_tags(self): return [Datatag.weights]
    class BadPass(TransformPass):
      def run(self, frame): pass

    # Normal registration
    DataManager.register(GoodPass)
    DataManager.register(GoodPass2)

    # Allow idempotency
    DataManager.register(GoodPass)
    DataManager.register(GoodPass2)

    # Allow deregistration and registration
    DataManager._deregister(GoodPass)
    DataManager._deregister(GoodPass2)

    # Don't allow empty invalidation tag lists
    with self.assertRaises(TypeError):
      DataManager.register(BadPass)

    # Allow instance-driven registration
    mgr = DataManager()
    mgr.register(GoodPass)


  def test_invalidation(self):
    mgr = DataManager()

    for tag in Datatag.get_all_tags():
      assert mgr._cache[tag] is None, 'Cache initialization failure for tag '+str(tag)

    # Check that pseudo-tags aren't actually in the cache.
    assert Datatag.ALL not in mgr._cache, 'Pseudo-tag found in cache.'
    assert Datatag.NONE not in mgr._cache, 'Pseudo-tag found in cache.'
    assert len(mgr._cache)==len(Datatag.get_all_tags()), 'Cache size mismatch'

    # Invalidate all
    for test_tag in Datatag.get_all_tags():
      mgr._cache[test_tag] = 123
    mgr.invalidate(Datatag.ALL)
    for test_tag in Datatag.get_all_tags():
      assert mgr._cache[test_tag] is None, 'Cache invalidation failure for tag '+str(tag)

    # Invalidate none
    for test_tag in Datatag.get_all_tags():
      mgr._cache[test_tag] = 123
    mgr.invalidate(Datatag.NONE)
    for test_tag in Datatag.get_all_tags():
      assert mgr._cache[test_tag]==123, 'Cache invalidation failure for tag '+str(tag)

    # Invalidate one
    for test_tag in Datatag.get_all_tags():
      mgr._cache[test_tag] = 123
    mgr.invalidate(Datatag.weights)
    for test_tag in Datatag.get_all_tags():
      if test_tag is Datatag.weights:
        assert mgr._cache[test_tag] is None, 'Cache invalidation failure for tag '+str(tag)
      else:
        assert mgr._cache[test_tag]==123, 'Cache invalidation failure for tag '+str(tag)

