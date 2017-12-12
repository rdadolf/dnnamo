#!/bin/bash

LINT_FLAGS=-rn

pylint $LINT_FLAGS nnmodel
pylint $LINT_FLAGS test
pylint $LINT_FLAGS tools
