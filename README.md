To run this code, download the
fast SSK implementation from
https://github.com/helq/python-ssk
and apply the patch in `python-ssk.patch`.
This is required for the StringKernel class.
Alternatively,
you may comment out the import of `string_kernel`
if you do not wish to run any experiments involving the string subsequence kernel.

# Documentation

If you simply wish to reproduce the experiments,
the main entry point is the file `run_exp_suite.py`.
This runs the majority of the experiments
described in the thesis, caching the results
to speed up future re-runs.

If you wish to understand the codebase
in depth, then we recommend reading the tests.
These help document the interfaces in the codebase,
and show how the different pieces fit together.
Alternatively, you could read `run_exp_suite.py`
as a starting point and then follow references from there.

The names of modules, classes,
and functions are designed to be self-documenting.
Doc strings are used in those places where we feel it helps
understand the design or purpose of specific parts of the code.
