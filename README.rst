project-skeleton
================

This project skeleton provides the starting point for your student project.

Getting Started
---------------
.. _miniconda: https://conda.io/docs/install/quick.html#linux-miniconda-install
First you need to install miniconda_ on your system.

Next, configure the settings.py file and set values for PROJECT, AUTHOR and EMAIL.

Then simply run :code:`make`, which will rename the project folder and create
an initial conda environment.

Having created the environment, start it up with

.. code-block:: shell

    source activate **project**

Make sure you replace **project** with the value you set for PROJECT in settings.py.

In order to test your setup, run :code:`make test` and :code:`make quality`.
They should both run OK.

You can also build the initial documentation with :code:`make doc` and view it with :code:`make view`.



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

In order to set up a new student project, fork the project to the admin namespace.

There change the project name and path for the new project.

Finally transfer the project to the ise-squad namespace.

Make sure the ise-squad-runner is enabled and merging is only possible after a successful build.