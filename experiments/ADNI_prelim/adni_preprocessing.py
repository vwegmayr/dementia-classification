from os import path
import argparse
import sys
import pickle

from dementia_prediction.config_wrapper import Config
from dementia_prediction.preprocessing.data_pipeline import DataPipeline

config = Config()

parser = argparse.ArgumentParser(description="Preprocess the ADNI data")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()

config.parse(path.abspath(args.paramfile))
paths = config.config.get('paths')
data_path = path.abspath(paths['data'])

with open(paths['ad_list'], 'rb') as filep:
    ad_aug_list = pickle.load(filep)
with open(paths['nc_list'], 'rb') as filep:
    nc_aug_list = pickle.load(filep)

# Initialise the Data preprocessing pipeline
pipeline = DataPipeline(in_folder=data_path,
                        params=config.config.get('parameters'))

# Augment ADNI AD images
if pipeline.rotate(regex=r"_mni_aligned\.nii\.gz",
                   split_on='_mni_aligned.nii.gz',
                   in_folder=paths['in_folder'],
                   out_folder=paths['out_folder_ad'],
                   angle=3,
                   patient_list=ad_aug_list):
    print("Augmentation of ADNI AD images successful.")
else:
    print("Error during ad augmentation.")

# Augment ADNI NC images
if pipeline.translate(regex=r"_mni_aligned\.nii\.gz",
                   split_on='_mni_aligned.nii.gz',
                   in_folder=paths['in_folder'],
                   out_folder=paths['out_folder_nc'],
                   pixels=4,
                   patient_list=nc_aug_list):
    print("Augmentation of ADNI NC images successful.")
else:
    print("Error during nc augmentation.")