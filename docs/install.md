# Prerequisites
You'll need to install some software separately before building Dnnamo.

The Dnnamo dependencies are kept in the top-level `requirements.txt` file.
You can use pip to install these automatically:

```bash
pip install --upgrade -r requirements.txt
```

If you're planning on using one of the supported external workload suites, the requirements for those are in the `models` directory.
Use pip the same way to install these requirements.

# Running tests

Dnnamo uses pytest for testing, but it contains two shortcut scripts in the `build` directory.

```bash
./build/test-summary.sh
```
The summary script gives a short overview of all tests, omitting any output or logging. This is useful if you expect the tests to pass (such as on a clean installtion). If a test fails, it does not return output, so use the other script.

```bash
./build/test.sh
```

This runs testing in normal mode, where all output and tracebacks are captured when a test fails. Note that the script also passes through command-line arguments, so if you want to run a single test or drop into the debugger on failure, you can specify that here (the name of the test file or `--pdb`, respectively). These are pytest options.

# Building documentation

Make sure you have `mkdocs` installed, then run:

```bash
./build/build_docs.sh
```

The documentation is placed in the `deploy` directory (created, if it doesn't exist).
Note that this documentation can be used off-line, too.
Just point a browser to the `deploy/index.html`
