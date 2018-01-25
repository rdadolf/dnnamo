import unittest

from dnnamo.core.amopass import TransformPass
from dnnamo.core.datamanager import Datatag, DataManager

class GoodPass(TransformPass):
  def run(self, frame): pass
  @property
  def invalidation_tags(self): return [Datatag(None,None,None,None)]

class GoodPass2(TransformPass):
  def run(self, frame): pass
  @property
  def invalidation_tags(self): return [Datatag('weights','all','all','all')]

class BadPass(TransformPass):
  def run(self, frame): pass
  # no invalidation tags

class TestDataManager(unittest.TestCase):
  def test_instantiation(self):
    _ = DataManager

  def test_registration(self):
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

    for tag in Datatag.all():
      assert mgr._cache[tag] is None, 'Cache initialization failure for tag '+str(tag)

    # Invalidate all
    for test_tag in Datatag.all():
      mgr._cache[test_tag] = 123
    mgr.invalidate(Datatag('all','all','all','all'))
    for test_tag in Datatag.all():
      assert mgr._cache[test_tag] is None, 'Cache invalidation failure for tag '+str(tag)

    # Invalidate none
    for test_tag in Datatag.all():
      mgr._cache[test_tag] = 123
    mgr.invalidate(Datatag('none','all','all','all'))
    for test_tag in Datatag.all():
      assert mgr._cache[test_tag]==123, 'Cache invalidation failure for tag '+str(tag)

    # Invalidate one
    for test_tag in Datatag.all():
      mgr._cache[test_tag] = 123
    mgr.invalidate(Datatag('weights','all','all','all'))
    for test_tag in Datatag.all():
      if test_tag.name=='weights':
        assert mgr._cache[test_tag] is None, 'Cache invalidation failure for tag '+str(tag)
      else:
        assert mgr._cache[test_tag]==123, 'Cache invalidation failure for tag '+str(tag)

