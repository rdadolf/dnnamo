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
    AnalysisManager._register(GoodExampleAnalysis)
    AnalysisManager._register(AnotherGoodExampleAnalysis)

    # Allow idempotency
    AnalysisManager._register(GoodExampleAnalysis)
    AnalysisManager._register(AnotherGoodExampleAnalysis)

    # Allow deregistration and registration
    AnalysisManager._deregister(GoodExampleAnalysis)
    AnalysisManager._deregister(AnotherGoodExampleAnalysis)

    # Don't allow empty invalidation tag lists
    with self.assertRaises(TypeError):
      AnalysisManager._register(BadExampleAnalysis)


  def test_invalidation(self):
    pass
