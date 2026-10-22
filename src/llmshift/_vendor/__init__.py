"""Third-party code shipped inside llmshift, so it installs without compiling anything.

yaml/  PyYAML 6.0.3, pure-Python modules from the official sdist
       (pyyaml-6.0.3.tar.gz, sha256 d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f),
       copied unchanged except that cyaml.py (bindings to the optional C extension) is left out.
       MIT License, see yaml/LICENSE. llmshift only uses SafeLoader and safe_dump, which are
       pure Python in PyYAML anyway; the PyPI package needs a C compiler on Pythons it has no
       wheels for (3.15, free-threaded builds).
"""
