import os.path
import unittest

from ..util import in_temporary_directory

from dnnamo.core.estimator import EstimatorIO
from dnnamo.core.primop import PrimopTypes

class TestEstimatorIO(unittest.TestCase):
  def test_instantiate(self):
    _ = EstimatorIO()
    _ = EstimatorIO({'hadamard': [([1,2,3],1)]})

  def test_builtins(self):
    e = EstimatorIO( {
      'hadamard': [ ([1,2,3],1), ([2,4,6],2), ([3,6,9],3) ],
      'dot': [ ([1,2,3],10), ([2,4,6],20), ([3,6,9],30) ],
      } )

    assert len(e['hadamard'])==3
    assert len(e['dot'])==3

    e['hadamard'].append( ([4,8,12],4) )
    assert len(e['hadamard'])==4

    e['hadamard'].extend([ ([5,10,15],5) ])
    assert len(e['hadamard'])==5

    assert e['dot'][0][1] == 10
    e['dot'] = [ ([1,2,3],100), ([2,4,6],200), ([3,6,9],300) ]
    assert e['dot'][0][1] == 100


  def test_io(self):
    FILENAME='test_estimatorIO_file'
    with in_temporary_directory() as d:
      e = EstimatorIO( {
        'hadamard': [ ([1,2,3],1), ([2,4,6],2), ([3,6,9],3) ],
        'dot': [ ([1,2,3],10), ([2,4,6],20), ([3,6,9],30) ],
        } )
      assert not os.path.exists(FILENAME)
      e.write(FILENAME)
      assert os.path.exists(FILENAME)

      e2 = EstimatorIO()
      e2.read(FILENAME)

      assert len(e)==len(e2)
      for k in e2:
        assert len(e[k])==len(e2[k])
        for lhs,rhs in zip(e[k],e2[k]):
          assert lhs[0]==rhs[0]
          assert lhs[1]==rhs[1]

  def test_mangled_data(self):
    with self.assertRaises( (TypeError) ):
      _ = EstimatorIO( [1,2,3] ) # Completely wrong data type
    with self.assertRaises( (TypeError) ):
      _ = EstimatorIO( {1:2, 3:4} ) # Keys and value wrong types
    with self.assertRaises( (TypeError) ):
      _ = EstimatorIO( {1: [ ([1,2],1) ] } ) # Key wrong type
    with self.assertRaises( (KeyError) ):
      _ = EstimatorIO( {'nonsense': [ ([1,2],1) ] } ) # Invalid op name
    with self.assertRaises( (TypeError) ):
      _ = EstimatorIO( {'dot': [ 4 ] } ) # Operator data not a pair
    with self.assertRaises( (IndexError) ):
      _ = EstimatorIO( {'dot': [ 'not_a_pair' ] } ) # iterable, but not a pair
    with self.assertRaises( (TypeError) ):
      _ = EstimatorIO( {'dot': [ (1,2) ] } ) # "1" is not an argument list
    
