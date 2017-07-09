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
ref_path = '/home/rams/4_Sem/Thesis/Data/NIFTI_1/CON018/DTI_avg_study_template.nii.gz'
#ref_path = '/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
# Initialise the Data preprocessing pipeline
pipeline = DataPipeline(in_folder=data_path,
                        params=config.config.get('parameters'))
DTI = 'DTI_MD'

if pipeline.eddy_correction():
    print("Eddy corrected")
if pipeline.brain_extraction(regex=r'-DTI_eddy_corrected\.nii\.gz$',
                             split_on='.nii.gz', bias=False):
    print("Extraction of brain from DTI is successful.")
else:
    print("Error extracting brain images from DTI.")

if pipeline.calculate_tensor(
        out_folder='/home/rams/4_Sem/Thesis/Data/DTI_data/'):
    print("Successful")

# Align the MO images to the T1 images for a patient
if pipeline.DTI_registration(regex=r"-DTI_MD\.nii\.gz$", ref_path='T1',
                             suffix='_T1_aligned.nii.gz',
                             in_folder='/home/rams/4_Sem/Thesis/Data/NIFTI/',
                             out_folder='/home/rams/4_Sem/Thesis/Data/DTI_MD_T1'
                                        '/'):
    print("Successful")
# Align the T1 aligned images to a specific subject
if pipeline.DTI_registration(regex=r"-DTI_MD_T1_aligned\.nii\.gz$",
                             ref_path='/home/rams/4_Sem/Thesis/Data/DTI_MD_T1/'
                                      'CON018-DTI_MD_T1_aligned.nii.gz',
                             suffix='_subject.nii.gz',
                             in_folder='/home/rams/4_Sem/Thesis/Data/DTI_MD'
                                       '_T1/',
                             out_folder='/home/rams/4_Sem/Thesis/Data/DTI_MD_subject/'
                             ):
    print("Successful")
# Find the average study specific template
if pipeline.average_template(regex=r"-DTI_MD_T1_aligned_subject\.nii\.gz$",
                             in_folder='/home/rams/4_Sem/Thesis/Data/DTI_MD'
                                       '_subject/',
                             out_folder='/home/rams/4_Sem/Thesis/Data/DTI_MD_avg'
                                        '/',
                             avg_filename='DTI_MD_avg_template.nii.gz'):
    print("Successful")
# Align the images to the study specific template
if pipeline.DTI_registration(regex=r"-DTI_MD_T1_aligned\.nii\.gz$",
                             ref_path='/home/rams/4_Sem/Thesis/Data/DTI_MD_avg/'
                                      'DTI_MD_avg_template.nii.gz',
                             suffix='_average.nii.gz',
                             in_folder='/home/rams/4_Sem/Thesis/Data/DTI_MD_T1/',
                             out_folder='/home/rams/4_Sem/Thesis/Data/DTI_MD_avg/'
                             ):
    print("Successful")


# Align the MO images to the T1 images for a patient
if pipeline.DTI_registration(regex=r"-CBF\.nii\.gz$", ref_path='T1',
                             suffix='_T1_aligned.nii.gz',
                             in_folder='/home/rams/4_Sem/Thesis/Data/NIFTI/',
                             out_folder='/home/rams/4_Sem/Thesis/Data/CBF_T1'
                                        '/'):
    print("Successful")

# Align the T1 aligned images to a specific subject
if pipeline.DTI_registration(regex=r"-CBF_T1_aligned\.nii\.gz$",
                             ref_path='/home/rams/4_Sem/Thesis/Data/CBF_T1/'
                                      'CON018-CBF_T1_aligned.nii.gz',
                             suffix='_subject.nii.gz',
                             in_folder='/home/rams/4_Sem/Thesis/Data/CBF'
                                       '_T1/',
                             out_folder='/home/rams/4_Sem/Thesis/Data/CBF_subject/'
                             ):
    print("Successful")
# Find the average study specific template
if pipeline.average_template(regex=r"-CBF_T1_aligned_subject\.nii\.gz$",
                             in_folder='/home/rams/4_Sem/Thesis/Data/CBF'
                                       '_subject/',
                             out_folder='/home/rams/4_Sem/Thesis/Data/CBF_avg'
                                        '/',
                             avg_filename='CBF_avg_template.nii.gz'):
    print("Successful")

# Align the images to the study specific template
if pipeline.DTI_registration(regex=r"-"+DTI+"_T1_aligned\.nii\.gz$",
                             ref_path='/home/rams/4_Sem/Thesis/Data/'+DTI
                                     +'_avg/'+DTI+'_avg_template.nii.gz',
                             suffix='_average.nii.gz',
                             in_folder='/home/rams/4_Sem/Thesis/Data/'+DTI\
                                              +'_T1_5/',
                             out_folder='/home/rams/4_Sem/Thesis/Data/'+DTI\
                                              +'_avg/'
                             ):
    print("Successful")


if pipeline.gaussian_smoothing(regex=
                               r"-"+DTI+"_T1_aligned_average\.nii\.gz$",
                            split_on= "_T1_aligned_average.nii.gz",
                            in_folder='/home/rams/4_Sem/Thesis/Data/'+DTI\
                                              +'_avg/',
                            out_folder='/home/rams/4_Sem/Thesis/Data/'+DTI\
                                              +'_smoothed/'
                            ):
    print("Successful")

if pipeline.subsample(regex=r"-"+DTI+"_smoothed\.nii\.gz$", split_on=
"_smoothed.nii.gz", in_folder='/home/rams/4_Sem/Thesis/Data/'+DTI\
                                              +'_smoothed/',
                      out_folder='/home/rams/4_Sem/Thesis/Data/'+DTI\
                                              +'_subsampled/'):
    print("Successful")
if pipeline.move(regex=r"-DTI_FA_subsampled\.nii\.gz$"):
    print("Successful")

if pipeline.rotate(regex=r"-"+DTI+"_subsampled\.nii\.gz$",
                   split_on="_subsampled.nii.gz",
                   in_folder='/home/rams/4_Sem/Thesis/Data/'+DTI\
                                              +'_subsampled/',
                   out_folder='/home/rams/4_Sem/Thesis/Data/'+DTI\
                                              +'_rot/',
                   angle=5):
    print("Successful")

if pipeline.rot_trans(regex=r"-"+DTI+"_subsampled\.nii\.gz$",
                      split_on="_subsampled.nii.gz",
                      in_folder='/home/rams/4_Sem/Thesis/Data/'+DTI \
                              + '_subsampled/',
                      out_folder='/home/rams/4_Sem/Thesis/Data/' + DTI \
                              + '_data_aug/'
                      ):
    print("Successful")
if pipeline.move(regex=r"-DTI_MO\.nii\.gz$",
                 out='/home/rams/4_Sem/Thesis/Data/DTI'):
    print("Successful")
