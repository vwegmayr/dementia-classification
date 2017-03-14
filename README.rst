project-skeleton
================

This project skeleton provides the starting point for your student project.

Clone the repository, go to the repo root directory and find + replace all appearances
of "AUTHOR" and "project_name".

Additionally, you have to modify the setup.py file according to your wishes.

Of course, this readme should be modified as well.

Tests and quality checks
------------------------

The main components of this skeleton are the test and quality checks.

In order to run the tests:

.. code-block:: shell

    make && make test

Run the code quality checks:

.. code-block:: shell

    make && make quality
    
Alternatively, you can run tests and quality checks from the repo root without docker:

.. code-block:: shell
    
    python -m unittest

    flake8 project_name
    pylint project_name
    
However, this assumes you have everything installed locally (e.g. flake8 & pylint).

Documentation
-------------
.. _sphinx: http://www.sphinx-doc.org/en/stable/
.. _readthedocs: http://docs.readthedocs.io/en/latest/index.html
This skeleton also includes the basic parts that one needs for documentation with
sphinx_ and readthedocs_ in the folder docs.

Python Packaging
----------------
.. _packaging: https://python-packaging.readthedocs.io/en/latest/
The skeleton is structured in a way which supports the creation of a python package.
Read more about packaging_.


Remarks for admin
----------------

In order to set up a new student project, fork the project to the ise-squad namespace.

There change the project name and path for the new project.

Make sure the ise-squad-runner is enabled and merging is only possible after a successful build.