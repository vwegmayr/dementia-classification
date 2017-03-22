project-skeleton
================

This project skeleton provides the starting point for your student project.

Getting Started
---------------
.. _miniconda: https://conda.io/docs/install/quick.html#linux-miniconda-install
First you need to install miniconda_ on your system.

Next, configure the settings.py file and set values for PROJECT, AUTHOR and EMAIL.

Then rename the project folder and create an initial conda environment with

.. code-block:: shell

    make folder && make env

Having created the environment, start it up with

.. code-block:: shell

    source activate project

Make sure you replace **project** with the value you set for PROJECT in settings.py.

In order to test your setup, run

.. code-block:: shell

    make test && make quality

They should both run OK.

You can also build and view the initial documentation with

.. code-block:: shell

    make doc && make view
    
The inital documentation also tries to give some insight into using sphinx.

Examples
--------
In addition to the basic functionality outlined above, the skeleton provides a couple of examples:

`doctest <project/examples/doctest.py>`_

`docstrings <project/examples/sphinx.py>`_

`estimator <project/examples/estimator.py>`_

`sphinx <doc/example_templates>`_

`config file <project/examples/example_config.yaml>`_
