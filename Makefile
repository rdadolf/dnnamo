# This is a slightly less horrible file than the last one, but still
# should be replaced with something better.
# Dependencies are not respected.

default:
	@echo 'Usage:'
	@echo '  --- Tests ---'
	@echo '  make test : run all the nnmodel library tests'
	@echo '  --- Modeling ---'
	@echo '  make ubench : build all microbenchmarks'
	@echo '  --- Documentation ---'
	@echo '  make docs : builds browsable documentation'
	@echo '  make docserver : runs a local web server to read the documentation'

.PHONY: test lint ubench docs

test: ubench
	build/run-nosetests.sh

lint: 
	build/run-linter.sh

ubench:
	@$(MAKE) -C nnmodel/devices

docs: docs/*.md
	build/build-docs.sh

docserver: docs
	mkdocs serve
