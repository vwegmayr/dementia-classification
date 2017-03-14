project-skeleton
================

This project skeleton provides the starting point for your student project.

Clone the repository, go to the repo root directory and find + replace all appearances
of "AUTHOR" and "project_name".

Additionally, you have to modify the setup.py file according to your wishes.

Before doing anything you need to build the container:

.. code-block:: shell

    make


**Before running any of the following commands you need to build the container first**


Run the tracking:

.. code-block:: shell

    make run


Running tests requires a TTY(hint: you have one, the gitlab-ci server does not):

.. code-block:: shell

    make test


To run the test without a TTY(as for example on the gitlab-ci server):

.. code-block:: shell

    make test_server


Run the code quality control tools:

.. code-block:: shell

    make quality

Remarks for admin
----------------

In order to set up a new student project, fork the project to the ise-squad namespace.

There change the project name and path for the new project.

Make sure the ise-squad-runner is enabled.