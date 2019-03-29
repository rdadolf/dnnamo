#!/bin/bash

LINT_FLAGS=-rn

pylint $LINT_FLAGS dnnamo && \
pylint $LINT_FLAGS test && \
pylint $LINT_FLAGS tools
