import unittest

from dnnamo.frameworks.tf import TFFramework
from dnnamo.loaders.tf import FathomLoader

try:
  import fathom
  _FATHOM = True
except ImportError:
  _FATHOM = False


@unittest.skipUnless(_FATHOM, 'No Fathom module found.')
class TestFathomLoader(unittest.TestCase):
  _models = ['Seq2Seq', 'MemNet', 'Speech', 'Autoenc', 'Residual', 'VGG', 'AlexNet', 'DeepQ' ]

  def test_seq2seq(self):
    TFFramework().load(FathomLoader, 'Seq2Seq')
  def test_memnet(self):
    TFFramework().load(FathomLoader, 'MemNet')
  def test_speech(self):
    TFFramework().load(FathomLoader, 'Speech')
  def test_autoenc(self):
    TFFramework().load(FathomLoader, 'Autoenc')
  def test_residual(self):
    TFFramework().load(FathomLoader, 'Residual')
  def test_vgg(self):
    TFFramework().load(FathomLoader, 'VGG')
  def test_alexnt(self):
    TFFramework().load(FathomLoader, 'AlexNet')
  def test_deepq(self):
    TFFramework().load(FathomLoader, 'DeepQ')
