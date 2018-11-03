import numpy as np
import unittest

from ..util import in_temporary_directory

from dnnamo.core.features import Features

class TestFeatures(unittest.TestCase):
  def test_instantiate(self):
    _ = Features()

  def test_add(self):
    f = Features()
    f.append([1,2,3], 4)
    f.append(np.array([1,2,3]), np.array(4))
    assert len(f.measurements)==2
    f2 = Features()
    f2.extend([[1,2,3],[4,5,6]], [7,8])
    assert len(f2.measurements)==2
    f2.concatenate(f)
    assert len(f2.measurements)==4

  def test_accessors(self):
    f = Features()
    f.append([1,2,3], 10)
    f.append([4,5,6], 11)
    op_args = f.op_arguments
    meas = f.measurements
    assert op_args.shape==(2,3)
    assert meas.shape==(2,)
    assert meas[0]==10 and meas[1]==11

  def test_invalid_shapes(self):
    f = Features()
    with self.assertRaises(AssertionError):
      f.append(1, 0)
    with self.assertRaises(AssertionError):
      f.append([1,2,3],[4,5,6])

  def test_argument_length_mismatch(self):
    f = Features()
    f.append([1,2,3], 0)
    with self.assertRaises(AssertionError):
      f.append([1,2,3,4], 0)
    with self.assertRaises(AssertionError):
      f.append([1,2], 0)
    f.append([4,5,6], 0)

  def test_io(self):
    f = Features()
    f.append([1,2,3], 10)
    f.append([2,3,4], 11)
    f.append([3,4,5], 12)

    with in_temporary_directory() as d:
      f.write('test_file')
      f2 = Features()
      f2.read('test_file')

    assert np.all(f.array==f2.array)
