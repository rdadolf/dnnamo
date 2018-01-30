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
    
class TestFramework(unittest.TestCase):
  def test_instantiation(self):
    frame = ExampleFramework()

# NOTE: pytest is apparently unable to parameterize (yes, that's how it's
#   ACTUALLY spelled, pytest people) class methods if the class is a
#   subclass of unittest.TestCase. This has something to do with the
#   invocation inside TestCase calling methods with no arguments.
#   So for this case, we just use a normal class (with test prefixes).
@pytest.mark.parametrize('mode', ['training', 'inference'])
@pytest.mark.parametrize('name,scope,ops', [
  ('graph','static','native'),
  ('graph','static','primitive'),
  ('graph','dynamic','primitive'),
  ('graph','dynamic','primitive'),
  ('weights','static','native'),
  ('timing','dynamic','native'),
  ('timing','dynamic','primitive'),
  ('ivalues','dynamic','native'),
])
# Eventually these should all pass, but for now many are unimplemented. Check
# for unexpected exceptions, but let the unimplemented ones fail gracefully.
@pytest.mark.xfail(raises=NotImplementedError)
class TestFrameworkCollectors(object):
  def test_collect(self, name, mode, scope, ops):
    frame = ExampleFramework()
    frame.load(RunpyLoader, 'test/test_models/simple_nnet.py')
    method = getattr(frame,'get_'+name)
    arg_keys = inspect.getargspec(method).args
    kwargs = {'mode':mode, 'scope':scope, 'ops':ops}
    kw = {k:v for k,v in kwargs.items() if k in arg_keys}
    v = method(**kw)
    assert v is not None, 'No data returned from '+str(method)+': '+str(v)

