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

    source activate project

Make sure you replace **project** with the value you set for PROJECT in settings.py.

In order to test your setup, run :code:`make test` and :code:`make quality`.
They should both run OK.

You can also build the initial documentation with :code:`make doc` and view it with :code:`make view`.

Examples
--------
In addition to the basic functionality outlined above, the skeleton provides a couple of examples:
`doctest <project/examples/doctest.py>`_