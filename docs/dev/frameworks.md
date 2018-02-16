# Frameworks

Dnnamo standardizes the interface to neural networks by wrapping them in model objects, but most of the heavy lifting internally is done using *frameworks*.
Frameworks provide a set of tools for interacting with models that leverage the underlying machine learning framework which the model is implemented in (hence the name).

# Interface

## Basics

- `load(loader, identifier, **kwargs)`: Loads a model. The `loader` argument specifies which [Dnnamo loader](/dev/loaders/index.html) to use to ingest a model. The `identifier` argument sets the name or path of the model to be loaded, but the exact meaning depends on the loader selected. Some loaders also have additional required or optional arguments, which can be supplied as keyword arguments to this method. They will be passed through untouched.

- `model`: *(property)* Returns the currently loaded Dnnamo model object.

## Data access

All data access methods use a subset of the same set of three arguments: `mode`, `scope`, and `ops`. These arguments provide different ways of collecting data. The `mode` option selects between inspecting the training and inference paths of a neural network. The `scope` argument chooses whether the data should reflect the computational graph as declared by the user or as run by the underlying framework (these can be different, mostly do to the effect of on-demand dataflow computation and graph optimization). The `ops` argument determines whether the resulting data is described in terms of native operation types (which mimic those suppied by the library) or [primitive operations](/dev/primops/index.html).

Not every accessor takes every argument, because some of them are nonsensical for the method in question.
For example, timing information cannot be collected on a static graph (we can *estimate* timing, but we can't *measure* it without running the graph).

- `get_graph(mode, scope, ops)`: 

- `get_timing(mode, ops)`: 

- `get_weights(mode)`: *Not Implemented Yet*

- `get_ivalues(mode)`: *Not Implemented Yet*
