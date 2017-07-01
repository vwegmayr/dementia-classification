import pickle
import random
import nibabel as nb
import re
import subprocess
import os
from scipy.ndimage.interpolation import shift

input_folder = '/local/adni_mni_smoothed/'
output_folder_ad = '/local/adni_augmented/ad/'
output_folder_nc = '/local/adni_augmented/nc/'
aug_list_folder = '/local/adni_normalized/'
with open(aug_list_folder+'adni_ad_aug_data2.pkl', 'rb') as filep:
    ad_aug_list = pickle.load(filep)
with open(aug_list_folder+'adni_nc_aug_data2.pkl', 'rb') as filep:
    nc_aug_list = pickle.load(filep)

# Augmenting AD images - Augment by rotating each image along x, y and z axis with
# a random angle between [-3, 3]
split_on = '_mni_aligned.nii.gz'
regex = r"_mni_aligned\.nii\.gz"
ctr = 0
for directory in os.walk(input_folder):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        if re.search(regex, input_file):
            patient_code = input_file.rsplit('/', 1)[1]
            patient_code = patient_code.split(split_on)[0]
            if patient_code in ad_aug_list:
                for direction in ['x', 'y', 'z']:
                    x_axis = y_axis = z_axis = 0
                    if direction == 'x':
                        x_axis = 1
                    if direction == 'y':
                        y_axis = 1
                    if direction == 'z':
                        z_axis = 1

                    output_file = output_folder_ad + '{0}_{1}_rot_mni_aligned.nii.gz'. \
                        format(patient_code, direction)
                    rot_matrix = 'rot_{0}.mat'.format(direction)
                    angle_rot = random.uniform(-3, 3)
                    print("Rotating image: " + input_file)
                    subprocess.call('makerot -c 45,55,45 -a {1},{2},'
                                    '{3} -t {4} -o {5}'
                                    .format(input_file,
                                            x_axis, y_axis, z_axis,
                                            angle_rot, rot_matrix),
                                    shell=True)
                    subprocess.call('flirt -in {0} -ref {1} -out '
                                    '{2} -applyxfm -init {3}'
                                    .format(input_file, input_file,
                                            output_file,
                                            rot_matrix),
                                    shell=True)
                    ctr += 1
print("Rotated ", ctr, " images")
ctr = 0
# NC data augmentation - translate the images in a random direction
for directory in os.walk(input_folder):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('/', 1)[1]
            pat_code = pat_code.split(split_on)[0]
            if pat_code in nc_aug_list: 
                output = output_folder_nc + pat_code+'_trans_mni_aligned.nii.gz'
                if not os.path.exists(output):
                    mri_image = nb.load(input_file)
                    aff = mri_image.get_affine()
                    mri_image = mri_image.get_data()
                    direction = random.randint(1,3)
                    shift_axis = [4,0,0]
                    if direction == 1:
                        shift_axis = [4,0,0]
                    elif direction == 2:
                        shift_axis = [0,4,0]
                    else:
                        shift_axis = [0,0,4]
                    translated_image = shift(mri_image, shift_axis, mode='nearest')
                    im = nb.Nifti1Image(translated_image, affine=aff)
                    nb.save(im, output)
                    print("Saving to "+output)
                    ctr += 1
                else:
                    print("Exists"+output)
print("Translated ", ctr, "images")
