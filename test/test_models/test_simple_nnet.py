# This file exists to help debug loading and import problems with test models.

import unittest

from .empty_model import EmptyModel
from .empty_model import __dnnamo_loader__ as empty_loader

from .simple_nnet import SimpleNNet
from .simple_nnet import __dnnamo_loader__ as simple_loader


class TestEmptyModel(unittest.TestCase):
  def test_instantiation(self):
    _ = EmptyModel()

  def test_loader(self):
    _ = empty_loader()

class TestSimpleNNet(unittest.TestCase):
  def test_instantiation(self):
    _ = SimpleNNet()

  def test_loader(self):
    _ = simple_loader()
