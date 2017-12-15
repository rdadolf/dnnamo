# Prerequisites
You'll need to install some software separately before building Dnnamo.

Dnnamo

  - Python 2.7
  - Numpy
  - nose (optional)
  - mkdocs (optional)

Tensorflow

  - bazel v1.5+
  - C++11 compiler
  - swig 3+
  - CUDA toolkit v7.0 (optional)
  - CuDNN v2 (optional)

# Building Tensorflow
To build a local copy of TensorFlow:

```bash
make configure
make tensorflow
make link
```

Note that the `make configure` step invokes Tensorflow's `configure` script, which asks a number of questions about your system configuration. If you run into trouble, consult the [Tensorflow installation documentation](https://www.tensorflow.org/versions/master/get_started/os_setup.html#installing-from-sources).

You'll need to point your `PYTHONPATH` to the `install/tensorflow` directory in order to pick up the copy of TensorFlow you just built.

# Running tests

Currently, Dnnamo uses `nose` for testing. Run:

```bash
make test
```

# Building documentation

Make sure you have `mkdocs` installed, then run:

```bash
make docs
```
