Dnnamo standardizes the interface to neural networks by wrapping them in model objects, but most of the heavy lifting internally is done using *frameworks*.
Frameworks provide a set of tools for interacting with models that leverage the underlying machine learning framework which the model is implemented in (hence the name).

# Interface

- `load(loader, identifier, **kwargs)`: Loads a model. The `loader` argument specifies which [Dnnamo loader](/dev/loaders/index.html) to use to ingest a model. The `identifier` argument sets the name or path of the model to be loaded, but the exact meaning depends on the loader selected. Some loaders also have additional required or optional arguments, which can be supplied as keyword arguments to this method. They will be passed through untouched.

- `graph`: *(property)* Returns a computational graph object. This is specific to the underlying library the loaded model is using.

- `model`: *(property)* Returns the currently loaded Dnnamo model object.
