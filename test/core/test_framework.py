import inspect
import pytest
import unittest
import itertools

from dnnamo.core.framework import Framework
from dnnamo.loader import RunpyLoader
from dnnamo.core.translator import Translator
from dnnamo.core.graph import DnnamoGraph

class ExampleTranslator(Translator):
  def translate(self, graph):
    # obviously this doesn't translate anything, but it's good enough for tests
    return DnnamoGraph()

class ExampleFramework(Framework):
  @property
  def translator(self):
    # Since the translator doesn't do anything, we can always return a new one
    return ExampleTranslator()

  @property
  def ExemplarRegistry(self):
    # Just need a non-abstract property implemented
    return None

  @property
  def SyntheticModel(self):
    # Just need a non-abstract property implemented
    return None
    
class TestFramework(unittest.TestCase):
  def test_instantiation(self):
    frame = ExampleFramework()

