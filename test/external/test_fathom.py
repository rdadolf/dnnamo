import unittest

from dnnamo.framework.tf import TFFramework
from dnnamo.loader import TFFathomLoader

try:
  import fathom
  _FATHOM = True
except ImportError:
  _FATHOM = False


@unittest.skipUnless(_FATHOM, 'No Fathom module found.')
class TestTFFathomLoader(unittest.TestCase):
  _models = ['Seq2Seq', 'MemNet', 'Speech', 'Autoenc', 'Residual', 'VGG', 'AlexNet', 'DeepQ' ]

  def test_seq2seq(self):
    TFFramework().load(TFFathomLoader, 'Seq2Seq')
  def test_memnet(self):
    TFFramework().load(TFFathomLoader, 'MemNet')
  def test_speech(self):
    TFFramework().load(TFFathomLoader, 'Speech')
  def test_autoenc(self):
    TFFramework().load(TFFathomLoader, 'Autoenc')
  def test_residual(self):
    TFFramework().load(TFFathomLoader, 'Residual')
  def test_vgg(self):
    TFFramework().load(TFFathomLoader, 'VGG')
  def test_alexnt(self):
    TFFramework().load(TFFathomLoader, 'AlexNet')
  def test_deepq(self):
    TFFramework().load(TFFathomLoader, 'DeepQ')
