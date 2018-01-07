Neural networks can be implemented in a variety of ways, so applying standard tools to them can sometimes be tricky.
Dnnamo works around this issue by requiring models to be wrapped in a lightweight interface which exposes standard features.

# Types of Models

Not all neural networks are created equal, and sometimes it can be difficult to access the full functionality of a model without large overhead.
For instance, a model integrated into a larger pipeline might be easy to export as a computational graph but difficult to run without the rest of the pieces.
Dnnamo provides several different interfaces to models to allow for these scenarios.
These model classes provide increasing levels of functionality and, in turn, more options for analysis, modeling, and optimization.

## Immutable

Immutable models provide read-only access to a neural network.
The class, `dnnamo.core.model.ImmutableModel`, exposes just two functions:

- `get_graph()`: returns an object corresponding to the computational graph.
- `get_weights(keys=None)`: returns a dictionary with all the model's weights, associated with a unique identifier. Optionally, you can pass in a list of keys. This will return a dictionary containing only the specified weights.

Both these functions are framework-dependent, so the return values will depend on how the neural network is implemented.

## Static

Static models add the ability to modify the model's current weights.
While it doesn't provide the ability to learn or infer anything, it can be useful, for instance, to analyze or adjust the distribution of weight values or perform static optimization like compression.

- `set_weights(kv)`: updated the model's weights according to a dictionary of identifier-array pairs. Weight whose identifiers exist in the model but not in the supplied argument will not be modified.

## Dynamic

Dynamic models provide full functionality.
In addition to being able to train and make predictions, dynamic models support more sophisticated tools, including optimizations which require retraining.
This requires three new functions:

- `run_train`: TBD
- `run_inference`: TBD
- `get_activations`: TBD

