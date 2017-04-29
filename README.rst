Dementia Risk Prediction
================

Ranked already as number six leading cause of death in the US, Dementia is of
major concern. Especially important is early diagnosis, even more so a scaleable
and reliable way to estimate a persons risk to develop the disease. One hope is
Magnetic Resonance Imaging (MRI) which provides a non-invasive means to examine
sensitive organs such as the brain. It is our goal to implement Dementia risk
prediction from MR images and discrimate the progressive patients from the stable
cognitive function patients by using and extending state-of-the-art methods from
deep learning.

Getting Started
---------------
To get the latest project code, clone the project 'develop' branch

.. code-block:: shell

    git clone -b develop git@gitlab.vis.ethz.ch:ise-squad/mri-fusion.git

.. _miniconda: https://conda.io/docs/install/quick.html#linux-miniconda-install
You need to install miniconda_ on your system.

Create an initial conda environment with

.. code-block:: shell

    make env

Having created the environment, start it up with

.. code-block:: shell

    source activate dementia_prediction

In order to test the setup,

.. code-block:: shell

    make test && make quality

They should both run OK.

You can also build and view the initial documentation with

.. code-block:: shell

    make doc && make view
    
Experiments
--------
Before running any experiments, the MR Images dataset need to be downloaded and preprocessed.
Run

.. code-block:: shell

    make data

The MR Images are downloaded to the local disk into a new 'Data' folder.

## Preprocessing

Preprocessing script will extract the brain from an MR Image and align all the extracted brains
to a study specific template. For this purpose, choose any one of the patients code as the reference
image in `ref_path` of `experiments/T1_preprocessing/params.yaml`. (By default it is CON018)

Preprocessing requires installation of FSL tool. Please follow the setup instructions
`here <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux/>`_ to download and install FSL.

Each image takes around 10minutes to complete the whole preprocessing pipeline. You can avoid
the preprocessing step if your data folder already contains the `-T1_brain_rotation_x/y/z.nii.gz`
images.

To preprocess the data, run

.. code-block:: shell

        python experiments/T1_preprocessing/t1_preprocess.py experiments/T1_preprocessing/params.yaml


## 3D CNN

Sumatra can be used to track the records of the experiments that are run on 3D CNN baseline model.
Setup sumatra with

.. code-block:: shell

       make smt

To try the 3D CNN model, run

.. code-block:: shell

	smt run -m experiments/T1_Baseline/t1_3dcnn.py experiments/T1_Baseline/params.yaml

The checkpoint and summary files can be viewed `here <http://192.33.91.83:9183/dementia_prediction/>`_ 
