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

#ref_path = '/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
# Initialise the Data preprocessing pipeline
pipeline = DataPipeline(in_folder=data_path,
                        params=config.config.get('parameters'))

if pipeline.brain_extraction(regex=r'-T1\.nii\.gz$',
                             split_on='.nii.gz', bias=True):
    print("Extraction of brain from T1_weighted MRI is successful.")
else:
    print("Error extracting brain images from T1_weighted MRI.")


if pipeline.linear_registration(ref_path, 0):
    print("Registered the brain images to a study specific template "
          "successfully.")
else:
    print("Error in registering extracted brain images to a study specific "
          "template")

if pipeline.gaussian_smoothing(regex= r"-T1_brain_avg_template_aligned\.nii\.gz$",
                                split_on="_avg_template_aligned.nii.gz",
                               in_folder='/home/rams/4_Sem/Thesis/Data/T1_brain_avg/',
                               out_folder='/home/rams/4_Sem/Thesis/Data/T1_brain_smoothed/'
                               ):
    print("Successful smoothing of the brain images with given gaussian "
          "parameters.")
else:
    print("Error in gaussian smoothing of brain images.")


pipeline.subsample(regex=r"-T1_brain_smoothed\.nii\.gz$",
                   split_on="_smoothed.nii.gz",
                   in_folder='/home/rams/4_Sem/Thesis/Data/T1_brain_smoothed/',
                   out_folder='/home/rams/4_Sem/Thesis/Data'
                              '/T1_brain_subsampled/'
                   )
pipeline.rotate(regex=r"-T1_brain_subsampled\.nii\.gz$",
                split_on="_subsampled.nii.gz",
                in_folder='/home/rams/4_Sem/Thesis/Data/T1_brain_subsampled/',
                out_folder='/home/rams/4_Sem/Thesis/Data/T1_brain_subsampled/',
                angle=5
                )
pipeline.rot_trans(regex=r"-T1_brain_subsampled\.nii\.gz$",
                split_on="_subsampled.nii.gz",
                   in_folder='/home/rams/4_Sem/Thesis/Data'
                             '/T1_brain_subsampled/',
                   out_folder='/home/rams/4_Sem/Thesis/Data'
                             '/T1_brain_subsampled/'
                   )
