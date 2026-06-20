* Every new module foo.py needs a pytest-compatible foo_test.py that exercises every function and method. Run each new test module and confirm that all tests in it pass.
* Python code formatting follows PEP-8.
* All new code gets PEP 484 and PEP 526 type annotation. New code needs to pass mypy type checking.
* New markdown documents live in docs/ unless otherwise requested.
* Functional tests live in tests/ and need not be pytest compatible.
* All pulldown menus use react-select.
* Check every change into git with useful comments.
* Every program in bin that generates a redis key should be added to the bin/rebuild_redis script.
* A missing package on production is a packaging error.
* Update docs/api-reference.md every time we change REST APIs.
* Priority order of CLI parameters is CLI --<parameter>, environment variable <PARAMETER>, config file (if we have one), and finally hardcoded default.
* Changes are made with TDD. Write tests first, get them confirmed by a human, and then make the test pass with an implementation.
* When failing tests are checked into git, they get pytest xfail tags so the head always has all tests passing--this allows bisect bug searches.
* A correllary of the last two constraints is that we need to implement bare skeletons of implementations so that there's enough to import--we can not xfail an import failure.
* Whenever we create a new CouchDB database, please update docs/couchdbs.md.
* Programs that are needed for one time fixes go in fixes/ rather than bin/.
* Please don't check credentials (especially passwords) into git.
* Keep wheel versions and their corresponding deb versions in sync.
