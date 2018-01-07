Because there are many ways to implement a neural net, there's no one way to capture them.
Machine learning frameworks sometimes provide externalizeable formats, which expose a varying degree of functionality.
Other times, a network is built into a software library and can only be used directly.
To solve this, Dnnamo provides *loaders*.
Loaders are bits of code that take in the location or name of a network and return a corresponding Dnnamo model object.
Depending on the loader and the implementation of the model, this object might be one of several [model types](/dev/models/index.html#types-of-models).

# Common

### `RunpyLoader`

RunpyLoader loads a model by running a Python function. Specifically, it imports either a file or a package and looks for the special function `__dnnamo_loader__`. This function should take no arguments and return a instance of a subclass of `dnnamo.core.model.BaseModel`.

Required arguments:

- `identifier`: The name of the Python file or module to load. If your model comes from a single file, use the file name without the `.py` extension (as you would if you were using the `import` keyword in Python normally).

Optional arguments:

- `pypath`: A list of paths which are temporarily added to `sys.path` for the purposes of loading the `identifier` module.

# TensorFlow

### `FathomLoader`

[Fathom](https://github.com/rdadolf/fathom) is a set of reference workloads written in TensorFlow. This loader imports the Fathom library and dynamically creates a Dnnamo model out of one of its workloads. **Note that these are currently `ImmutableModel`s only and do not yet support runtime information.**

Required arguments:

- `identifier`: The name of the Fathom model to load, all lowercase.

Optional arguments:

- `fathompath`: A single string containing the path to where the Fathom workloads are installed.

