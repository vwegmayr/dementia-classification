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

You can also build and view the initial documentation with

.. code-block:: shell

    make doc && make view
    
Experiments
--------
Install the package with

.. code-block:: shell

    pip install -e .


To setup sumatra

.. code-block:: shell

       make smt


The checkpoint and summary files can be viewed `here <http://192.33.91.83:9183/dementia_prediction/>`_

Preprocessing
------------

Raw images of any modality should be preprocessed first. The following preprocessing steps can be followed:

a. Using FSL bet with parameter f as 0.35, remove the skull and extract the brain from the raw image.
b. Align all the brain images to the standard MNI 152 2mm brain template.
c. Smooth the aligned brain images by applying Gaussian Smoothing with sigma 1mm.
d. Divide the dataset into training and validation data.
e. Normalize the dataset using the following three steps:

   a. Normalize each image individually to mean 0 and variance 1.
   b. Normalize each pixel across the dataset. For this purpose find the mean and variance of each pixel for
      the training data only to avoid double dipping.
   c. Normalize each image individually again to mean 0 and variance 1.

Model Training Tutorial
----------------------

For this tutorial, we will use the preprocessed toy data in ./Data/ folder.

Data conventions: 

UHG - University Hospitals Geneva [6 images each in training and validation data balanced across 2 classes]

ADNI - Alzeimers Disease Neuroimaging Iniative [6 images each in training and validation data balanced across 3 classes]


There are three modalities of data from each institute - T1, T2 and DTI FA

Models Conventions:

Baseline model - 3D CNN model trained on any single modality.

Fusion model - 3D CNN model trained by fusing baseline models.

Transfer finetuning model - 3D CNN model trained by transferring weights from another model and finetuned.

UHG T1 Baseline model:

.. code-block:: shell

       python experiments/Baseline/3dcnn.py experiments/Baseline/toy_parameters/UHG_T1_params.yaml

       
This model uses the UHG T1 data from ./Data/UHG_T1 directory and stores the baseline model in ./output/UHG_T1

UHG DTI FA Baseline model:

.. code-block:: shell

       python experiments/Baseline/3dcnn.py experiments/Baseline/toy_parameters/UHG_DTI_FA_params.yaml

This model uses the UHG DTI FA data from ./Data/UHG_DTI_FA directory and stores the baseline model in ./output/UHG_DTI_FA

UHG T2 Baseline model:

.. code-block:: shell

       python experiments/Baseline/3dcnn.py experiments/Baseline/toy_parameters/UHG_T2_params.yaml

This model uses the UHG T2 data from ./Data/UHG_T2 directory and stores the baseline model in ./output/UHG_T2

Fusion model:

.. code-block:: shell

       python experiments/multimodal/multimodal.py experiments/multimodal/toy_parameters/MNI_aligned_params.yaml

This model uses the models stored in ./output/UH_T2 ./output/UHG_T1 ./output/UHG_DTI_FA as fixed feature extractors
and then trains a fully connected layer on top of it and stores the model in ./output/UHG_multimodal/

## Transfer Learning

To improve the performance of the baselines and the fusion model, transfer learning is employed by using ADNI dataset.
For transfer learning, initially all the inidividual modality baselines are run and then the weights are transferred
to the UHG baselines and the UHG models are further finetuned.

ADNI T1 Baseline model:

.. code-block:: shell

       python experiments/Baseline/adni_3dcnn.py experiments/Baseline/toy_parameters/ADNI_T1_params.yaml

This model uses the ADNI T1 data from ./Data/ADNI_T1 directory and stores the baseline model in ./output/ADNI_T1

UHG T1 Transfer finetuning model:

.. code-block:: shell

       python experiments/transfer_learning/tl.py experiments/transfer_learning/toy_parameters/T1_params.yaml

This model uses the model stored in ./output/ADNI_T1 and finetunes using the data from ./Data/UHG_T1 and stores the
finetuned model at ./output/UHG_T1/transfer

UHG T2 Baseline model:

.. code-block:: shell

       python experiments/Baseline/adni_3dcnn.py experiments/Baseline/toy_parameters/ADNI_T2_params.yaml

This model uses the ADNI T2 data from ./Data/ADNI_T2 directory and stores the baseline model in ./output/ADNI_T2

UHG T2 Transfer finetuning model:

.. code-block:: shell

       python experiments/transfer_learning/tl.py experiments/transfer_learning/toy_parameters/T2_params.yaml

This model uses the model stored in ./output/ADNI_T2 and finetunes using the data from ./Data/UHG_T2 and stores the
finetuned model at ./output/UHG_T2/transfer

UHG DTI FA Baseline model:

.. code-block:: shell

       python experiments/Baseline/adni_3dcnn.py experiments/Baseline/toy_parameters/ADNI_DTI_FA_params.yaml

This model uses the ADNI DTI FA data from ./Data/ADNI_DTI_FA directory and stores the baseline model in ./output/ADNI_DTI_FA

UHG DTI FA Transfer finetuning model:

.. code-block:: shell

       python experiments/transfer_learning/tl.py experiments/transfer_learning/toy_parameters/DTI_FA_params.yaml

This model uses the model stored in ./output/ADNI_DTI_FA and finetunes using the data from ./Data/UHG_DTI_FA and stores the
finetuned model at ./output/UHG_DTI_FA/transfer

Fusing the transferred models:

The transferred and finetuned models can be fused as fixed feature extractors similar to fusing the individual baselines

.. code-block:: shell

       python experiments/multimodal/multimodal_fusion.py experiments/multimodal/toy_parameters/transfer_toptuning.yaml

This model uses the transferred finetuned models stored in ./output/UHG_T1/transfer ./output/UHG_T2/transfer ./output/UHG_DTI_FA/transfer
and trains a fully connected layer on top of it and stores the model in ./output/UHG_multimodal/transfer_toptuning/
Better performance is expected to be from finetuning this fusion model rather than toptuning.

