import nibabel as nb
import numpy as np
import pickle
import sys
import os
from os import path
import random
from nibabel.affines import apply_affine
import scipy.ndimage as snd
import scipy.ndimage.interpolation as sni
import argparse

from dementia_prediction.config_wrapper import Config

def reconstruct_image(image_path, filter):
    mri_image = nb.load(image_path)
    affine = mri_image.get_affine()
    mri_image = mri_image.get_data()
    mri_image = mri_image.flatten()
    filtered_image = np.asarray(filter)
    mri_image = filtered_image.reshape(91, 109, 91)
    rec_image = nb.Nifti1Image(mri_image, affine=affine)
    return rec_image

def construct_image_from_dic(dic):
    mri_image = np.asarray(dic)
    mri_image = mri_image.reshape(91, 109, 91)
    standard_image = nb.load('/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz')
    standard_affine = standard_image.get_affine()
    mri_image = nb.Nifti1Image(mri_image, affine=standard_affine)
    nb.save(mri_image, './constructed_image.nii.gz')
    load_image = nb.load('./constructed_image.nii.gz')

def translate_image(filename):
    pixels = random.uniform(-4, 4)
    mri_image = nb.load(filename)
    affine = mri_image.get_affine()
    mri_image = mri_image.get_data()
    mri_image = sni.shift(mri_image, [pixels, 0, 0], mode='nearest')
    trans_image = nb.Nifti1Image(mri_image, affine)
    nb.save(trans_image, './translated.nii.gz')
def rotate_image():
    standard_image = nb.load('/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz')
    standard_data = standard_image.get_data()
    affine = standard_image.get_affine()
    rot = sni.rotate(standard_data, -5, (2,0), reshape=False)
    rot_image = nb.Nifti1Image(rot, affine)
    nb.save(rot_image, './rotated.nii.gz')
    
def main():
    '''
    construct = True
    if construct == True:
        with open('./UHG_T2_data_mean.pkl', 'rb') as filep:
            image_dic = pickle.load(filep)
            construct_image_from_dic(image_dic)
    rotate_image()
    translate_image('/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz')
    sys.exit(0)
    '''

    config = Config()
    parser = argparse.ArgumentParser(description="Visualize features")
    parser.add_argument("paramfile", type=str,
                        help='Path to the parameter file')
    args = parser.parse_args()
    config.parse(path.abspath(args.paramfile))
    params = config.config.get('parameters')

    file_path = params['file_path']
    features_file = params['features_path']
    features = pickle.load(open(features_file, 'rb'))
    out_path = params['out_path']

    print(len(features))
    IMG_SIZE = params['height'] * params['width'] * params['depth']
    filter = np.zeros(IMG_SIZE)
    for index in features:
        filter[index] = 1
    rec_image = reconstruct_image(file_path, filter)

    nb.save(rec_image, out_path)

if __name__ == "__main__":
    main()
