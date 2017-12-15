#!/bin/bash

die() { echo "ERROR: $@"; exit 1; }

[ -d .git/hooks ] || die 'No .git/hooks directory found. Are you in the top-level dnnamo directory?'

cat > .git/hooks/pre-commit <<EOF
#!/bin/sh

# What is a pre-commit hook?
# (via Scott Chacon and Ben Straub's "Pro Git" book, http://git-scm.com/book)
#
# The pre-commit hook is run first, before you even type in a commit message.
# It’s used to inspect the snapshot that’s about to be committed, to see if
# you’ve forgotten something, to make sure tests run, or to examine whatever
# you need to inspect in the code. Exiting non-zero from this hook aborts the
# commit, although you can bypass it with git commit --no-verify. You can do
# things like check for code style (run lint or something equivalent), check
# for trailing whitespace (the default hook does exactly this), or check for
# appropriate documentation on new methods.

die() { echo "ERROR: $@"; exit 1; }
./build/run-linter.sh || die 'Linter reported errors. Aborting commit...'
EOF
chmod +x .git/hooks/pre-commit
