* Every new module foo.py needs a pytest-compatible foo_test.py that exercises every function and method. Run each new test module and confirm that all tests in it pass.
* Python code formatting follows PEP-8.
* All new code gets PEP 484 and PEP 526 type annotation. New code needs to pass mypy type checking.
* New markdown documents live in docs/ unless otherwise requested.
* Functional tests live in tests/ and need not be pytest compatible.
* All pulldown menus use react-select.
