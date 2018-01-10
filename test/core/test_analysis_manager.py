import unittest

from dnnamo.core.analysis import Analysis, AnalysisResult
from dnnamo.core.manager import AnalysisManager, InvalidationTag

class TestAnalysisManager(unittest.TestCase):
  def test_instantiation(self):
    _ = AnalysisManager

  def test_registration(self):
    class GoodExampleAnalysis(Analysis):
      @property
      def invalidation_tags(self):
        return [InvalidationTag.NONE]
    class AnotherGoodExampleAnalysis(Analysis):
      @property
      def invalidation_tags(self):
        return [InvalidationTag.GRAPH_STRUCTURE]
    class BadExampleAnalysis(Analysis):
      pass

    # Normal registration
    AnalysisManager.register(GoodExampleAnalysis)
    AnalysisManager.register(AnotherGoodExampleAnalysis)

    # Allow idempotency
    AnalysisManager.register(GoodExampleAnalysis)
    AnalysisManager.register(AnotherGoodExampleAnalysis)

    # Allow deregistration and registration
    AnalysisManager._deregister(GoodExampleAnalysis)
    AnalysisManager._deregister(AnotherGoodExampleAnalysis)

    # Don't allow empty invalidation tag lists
    with self.assertRaises(TypeError):
      AnalysisManager.register(BadExampleAnalysis)

    # Allow instance-driven registration
    mgr = AnalysisManager()
    mgr.register(GoodExampleAnalysis)


  def test_invalidation(self):
    class AllAnalysis(Analysis):
      @property
      def invalidation_tags(self):
        return [InvalidationTag.ALL]
    class GraphStructureAnalysis(Analysis):
      @property
      def invalidation_tags(self):
        return [InvalidationTag.GRAPH_STRUCTURE]
    class NeverAnalysis(Analysis):
      @property
      def invalidation_tags(self):
        return [InvalidationTag.NONE]
    AnalysisManager.register(AllAnalysis)
    AnalysisManager.register(GraphStructureAnalysis)
    AnalysisManager.register(NeverAnalysis)

    mgr = AnalysisManager()
    assert mgr._cache[AllAnalysis] is None, 'Cache initialization failure for newly registered analysis.'

    for cls,invalidation_tag,res in [
      (AllAnalysis, InvalidationTag.ALL, None),
      (AllAnalysis, InvalidationTag.WEIGHT_VALUES, None),
      (AllAnalysis, InvalidationTag.NONE, 123),
      (NeverAnalysis, InvalidationTag.ALL, None), # asking for ALL overrides NONE
      (NeverAnalysis, InvalidationTag.GRAPH_STRUCTURE, 123),
      (NeverAnalysis, InvalidationTag.NONE, 123),
      (GraphStructureAnalysis, InvalidationTag.ALL, None),
      (GraphStructureAnalysis, InvalidationTag.GRAPH_STRUCTURE, None),
      (GraphStructureAnalysis, InvalidationTag.WEIGHT_VALUES, 123),
      (GraphStructureAnalysis, InvalidationTag.NONE, 123),
    ]:
      mgr._cache[cls] = 123
      assert mgr._cache[cls]==123, 'Cache storage failure:'+str((cls,invalidation_tag,res))
      mgr.invalidate(invalidation_tag)
      assert mgr._cache[cls]==res, 'Cache storage failure:'+str((cls,invalidation_tag,res))
