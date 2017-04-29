from os import path
import argparse
import sys

from dementia_prediction.config_wrapper import Config
from dementia_prediction.preprocessing.data_pipeline import DataPipeline

config = Config()
param_file = sys.argv[1]
config.parse(path.abspath(param_file))
"""
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help="Path to the MRI Data Directory")
parser.add_argument('-r', '--ref', help="Reference MR Image location")
args = parser.parse_args()
"""
paths = config.config.get('paths')
data_path = path.abspath(paths['data'])
ref_path = path.abspath(paths['ref'])

# Initialise the Data preprocessing pipeline
pipeline = DataPipeline(in_folder=data_path,
                        params=config.config.get('parameters'))
if pipeline.brain_extraction():
    print("Extraction of brain from T1_weighted MRI is successful.")
else:
    print("Error extracting brain images from T1_weighted MRI.")

if pipeline.linear_registration(ref_path, 1):
    print("Registered the brain images to a study specific template "
          "successfully.")
else:
    print("Error in registering extracted brain images to a study specific "
          "template")

if pipeline.gaussian_smoothing():
    print("Successful smoothing of the brain images with given gaussian "
          "parameters.")
else:
    print("Error in gaussian smoothing of brain images.")
pipeline.subsample()
pipeline.rotate()
