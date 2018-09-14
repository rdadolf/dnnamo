import unittest

from dnnamo.framework.tf import TFFramework
from dnnamo.loader import TFFathomLiteLoader

try:
  import fathomlite
  _FATHOMLITE = True
except ImportError:
  _FATHOMLITE = False


@unittest.skipUnless(_FATHOMLITE, 'No Fathom module found.')
class TestTFFathomLiteLoader(unittest.TestCase):
  #_models = ['Seq2Seq', 'MemNet', 'Speech', 'Autoenc', 'Residual', 'VGG', 'AlexNet', 'DeepQ' ]
  _models = ['MemNet', 'Autoenc', 'Residual', 'VGG', 'AlexNet']

  def test_memnet(self):
    TFFramework().load(TFFathomLiteLoader, 'MemNet')
  def test_autoenc(self):
    TFFramework().load(TFFathomLiteLoader, 'Autoenc')
  def test_residual(self):
    TFFramework().load(TFFathomLiteLoader, 'Residual')
  def test_vgg(self):
    TFFramework().load(TFFathomLiteLoader, 'VGG')
  def test_alexnet(self):
    TFFramework().load(TFFathomLiteLoader, 'AlexNet')
