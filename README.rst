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
======================

We use the preprocessed data in /local/UHG/UHG_<modality>/, /local/ADNI/ and /local/OASIS folders.

Data conventions: 

UHG - University Hospitals Geneva 
ADNI - Alzeimers Disease Neuroimaging Iniative
AIBL - Australian Imaging, Biomarker & Lifestyle 
OASIS - Open Access Series of Imaging Studies 

There are three imaging modalities of data from UHG, ADNI, AIBL - T1, T2 and DTI FA and T1 image modality from OASIS

UHG and OASIS data are split into 10 folds. Each parameter file in the code uses first fold of the data.
Change the fold number from 1 to [2,10] to change the data set.

UHG/OASIS/ADNI+AIBL T1/T2/DTI FA Baseline models:
---------------------

.. code-block:: shell

       python experiments/main.py experiments/Baseline/<data_set_name>/<modality_name>.yaml

       
This command trains the Baseline 3D CNN model on a single image modality data and stores the model checkpoints
for each epoch at ../models/<data_set>/ folder.

ADNI Binary Classification Tasks (NC vs MCI, MCI vs AD, NC vs AD):
--------------------------

.. code-block:: shell

       python experiments/main_run.py experiments/Baseline/AIBL_ADNI/aibl_adni_t1_nc_ad_params.yaml

Multi layer perceptron with 2 layers:
----------------------

.. code-block:: shell

       python experiments/main_run.py experiments/Baseline/perceptron_params/<data_set_and_modality>.yaml

Ensembling over epochs for ADNI model:
-------------------------

.. code-block:: shell

       python experiments/Baseline/adni_3dcnn_ensemble.py experiments/Baseline/AIBL_ADNI/ensemble_params.yaml

Fusion model:
------------

.. code-block:: shell

       python experiments/main_run.py experiments/multimodal/CV_fusion/multimodal_cv_1.yaml

This code runs the multimodal fusion architecture. The individual image modality pipelines can be fused either at
the first convolutional layer (parameter: 'fusion_layer': 'conv1') or at the 7th convolutional layer
(parameter: 'fusion_layer': 'conv7'). This architecture can be trained from scratch with all the weights initialized
with modified Xavier init (parameters: 'multimodal': 'scratch', 'fusion': 'scratch') or with the weights from the 
individual image modality baseline models (parameter: 'multimodal': 'finetune', 'fusion': 'finetune'). To freeze these
weights and only toptune the top layers, use parameter: 'multimodal': 'toptune', 'fusion_layer': 'conv1' or 'conv7'

Ensembling model:
----------------

.. code-block:: shell

       python experiments/main_run.py experiments/multimodal/UHG_ensembling/ensemble_params.yaml

This code ensembles the predictions from the three imaging modalities. A 3D CNN is trained on each image modality
of a subject and the logits are averaged for the final prediction.


Multichannel model:
-----------------

.. code-block:: shell

       python experiments/main_run.py experiments/multimodal/UHG_multichannel/multichannel_1.yaml


This code runs the multichannel model where each image modality is given as input as a separate channel to the first layer
of 3D CNN.
       
Multitask model:
--------------

.. code-block:: shell

       python experiments/main_run.py experiments/multimodal/UHG_multitask/UHG_cv1.yaml

Transfer Learning
=================

To improve the performance of the baselines and the fusion model, transfer learning is employed by using ADNI dataset.

To freeze the weights and only train the top layers, run the below code:
At present, toptuning for transfer until 1st layer i.e., parameter: 'transfer': 'conv1' or 
until fully connected layer ('transfer': 'fullcn') are only supported.

.. code-block:: shell

       python experiments/main_run.py experiments/transfer_learning/CV_toptuning/<OASIS_or_UHG_modality>.yaml


For finetuning,  weights can be transferred from any number of layers from bottom to top with the parameter
'transfer_depth': <num> where <num> is 1 for 1st convolution layer and 8 for the last fully connected layer.

.. code-block:: shell

       python experiments/main_run.py experiments/transfer_learning/CV_finetuning/<OASIS_or_UHG_modality>.yaml


 
