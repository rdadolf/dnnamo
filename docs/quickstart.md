
## For tool users



## For tool writers

NNModel provides a powerful interface for analyzing and manipulating deep neural networks.
All of the tools distributed with NNModel are built using this same library interface.

Let's write a new tool that uses NNModel to count the number of times the most expensive computational operation is executed.
Like most tools, we will need two major pieces: a neural network description, and a device description.

### Ingesting a model description

Let's use the CIFAR-10 example from TensorFlow, which has a pre-built wrapper which comes with NNModel.
The code for loading this model is built-in, we just need to point to the right source file.

```python
import nnmodel
from nnmodel.frameworks import TensorflowFramework

frame = TFFramework()
frame.load('models/tf/cifar10')
```

And now we'll need a machine description.
We'll load up a generic device profile, which describes how primitive operations map to hardware.
Because we'll just be using the machine model to
Then we load a set of parameters specific to a particular machine and software stack.
Often these are empirically measured, but they can be generated synthetically, too.

```python
from nnmodel.devices import TF_CPU
cpu = TF_CPU()
```
