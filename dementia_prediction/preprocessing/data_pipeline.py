"""
This module provides the functions for different stages of the Data
preprocessing pipeline.
The major tool used for preprocessing is FSL 5.0.

References:
    S.M. Smith, M. Jenkinson, M.W. Woolrich, C.F. Beckmann, T.E.J.
    Behrens, H. Johansen-Berg, P.R. Bannister, M. De Luca, I. Drobnjak,
    D.E. Flitney, R. Niazy, J. Saunders, J. Vickers, Y. Zhang, N. De Stefano,
    J.M. Brady, and P.M. Matthews. Advances in functional and structural MR
    image analysis and implementation as FSL. NeuroImage, 23(S1):208-19, 2004
"""

import os
import re
import subprocess
import random
import nibabel as nb
import pickle
from scipy.ndimage.interpolation import shift
import sys


class DataPipeline:
    """
    Instantiate this class with the MR Images data folder path and a
    dictionary of parameters for the different stages of the pipeline.
    The results of the data preprocessing are stored in the same folders as
    the raw images.
    File conventions used with an example:
        Raw Image: CON018-T1.nii.gz
        Brain Extracted Image: CON018-T1_brain.nii.gz
        Reference Subject Aligned Image: CON018-T1_brain_subject_aligned.nii.gz
        Average Study Template Aligned Image:
            CON018-T1_brain_avg_template_aligned.nii.gz
        Gaussian Smoothed Image: CON018-T1_brain_smoothed.nii.gz
    """

    def __init__(self, in_folder, params):
        """

        Args:
            in_folder: Path to the input MR Images
            params: Parameters for the Data preprocessing pipeline

        """
        self.bet = params['bet']
        self.registration = params['registration']
        self.input_folder = in_folder

    def eddy_correction(self):
        for directory in os.walk(self.input_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with "-DTI.nii.gz"
                input_file = os.path.join(directory[0], file)

                if re.search(r'-DTI\.nii\.gz$', input_file):
                    output_file = '{0}_eddy_corrected.nii.gz'. \
                        format(input_file.split('.nii.gz')[0])

                    if not os.path.exists(output_file):
                        print("Eddy correcting DTI..: " + input_file)
                        subprocess.call('eddy_correct {0} {1} 0 trilinear -v'
                                        .format(input_file,
                                                output_file),
                                        shell=True)
                        print('Generated eddy corrected: ' + output_file)
                    else:
                        print("File already exists: " + output_file)
        return True


    def brain_extraction(self, regex, split_on, bias, dict=0, dict_path='./'):
        """
         This function extracts the brain from the raw T1 weighted MR Images
         using bet tool of FSL 5.0.
         It might take 5-6 minutes for an image of ~7MB.
         Run this function parallelly with different file paths to get
         optimal performance.
        """
        ctr = 0
        for directory in os.walk(self.input_folder):
            # Walk inside the directory
            for file in directory[2]:
                input_file = os.path.join(directory[0], file)
                # If there is a dictionary of patients, extract brains of
                # patients only present in the dictionary since bet is
                # expensive
                if re.search(regex, input_file):
                    extract = 1
                    if dict == 1:
                        # This code is specific to ADNI file path
                        fp = open(dict_path, 'rb')
                        pat_dict = pickle.load(fp)
                        patient = input_file.rsplit('/',1)[1]
                        pat_list = patient.split('_')[1:4]
                        patient_code = '_'.join(pat_list)
                        if patient_code in pat_dict:
                            extract = 1
                        else:
                            extract = 0
                    if extract == 1:
                        output_file = '{0}_brain.nii.gz'. \
                            format(input_file.split(split_on)[0])

                        if not os.path.exists(output_file):

                            print("Extracting brain from file: " + input_file)

                            cmd = 'bet {0} {1} -B -f {2} -m -v'.format(
                                            input_file, output_file,
                                            self.bet["frac_intens_thres"])
                            if bias is False:
                                cmd = 'bet {0} {1} -f {2} -m -v'.format(
                                    input_file, output_file,
                                    self.bet["frac_intens_thres"])
                            subprocess.call(cmd, shell=True)
                            print('Generated extracted brain: ' + output_file)
                            ctr += 1
                        else:
                            print("File already exists: " + output_file)
        print(ctr)
        return True

    def ASL_preprocess(self):
        for directory in os.walk(self.input_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with "-T1.nii.gz"
                input_file = os.path.join(directory[0], file)
                # for T1:
                # regex = r'-T1\.nii\.gz$'
                # for DTI:
                regex = r'-ASL\.nii\.gz$'
                if re.search(regex, input_file):
                    output_diff = '{0}_single_diff.nii.gz'. \
                        format(input_file.split('.nii.gz')[0])
                    output_diff_mean = '{0}_single_diff_mean.nii.gz'. \
                        format(input_file.split('.nii.gz')[0])

                    if not os.path.exists(output_diff):
                        print("Extracting diff mean from file: " + input_file)
                        subprocess.call('asl_file --data={0} --ntis=1 --iaf=tc '
                                        '--diff --out={1} '
                                        '--mean={2}'
                                        .format(input_file,
                                                output_diff,
                                                output_diff_mean),
                                        shell=True)
                        print('Generated diff and diff mean: ' + output_diff)
                    else:
                        print("File already exists: " + output_diff)
        return True

    def calculate_tensor(self, out_folder):
        for directory in os.walk(self.input_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with "-DT1.nii.gz"
                input_file = os.path.join(directory[0], file)
                regex = r'-DTI\.nii\.gz$'
                if re.search(regex, input_file):
                    # To store outputs in out_folder
                    '''
                    output_folder = out_folder
                    output_file_base = output_folder + input_file.rsplit('/',
                                                                       1)[1]
                    output_file_base = '{0}'. \
                        format(output_file_base.split('.nii.gz')[0])
                    '''
                    input_file_base = '{0}'. \
                        format(input_file.split('.nii.gz')[0])

                    # To store outputs in same folder
                    output_file_base = input_file_base

                    brain_mask = '{0}_eddy_corrected_brain_mask.nii.gz'.\
                                format(input_file_base)
                    grad_dir = '{0}.bvec'.format(input_file_base)
                    grad_val = '{0}.bval'.format(input_file_base)
                    #if not os.path.exists(output_file_base+'_FA.nii.gz'):
                    subprocess.call('dtifit -k {0} -o {1} -m {2} -r {3} -b {4}'
                                    .format(input_file,
                                            output_file_base,
                                            brain_mask,
                                            grad_dir,
                                            grad_val),
                                            shell=True)
                    print("Generated S0, FA, MD, L1 - L3, V1 - V3 for "
                          + output_file_base)
                    #else:
                    #   print(output_file_base+' vectors already exist')
        return True

    def DTI_registration(self, regex, ref_path, suffix, in_folder, out_folder):
        """
        This function registers each image in 'input_folder' to an average
        study template.
        Initially, any specific subject can be chosen as reference image to
        align all MR Images. The output aligned images are averaged to
        generate a study specific template to which all the MR Images are
        again re-registered. This removes the bias in selecting a specific
        subject as reference image.


        Args:
            ref_path: Reference subject Brain MR Image file path for initial
                      registration
            iteration: Number of iterations for re-registration on the
            generated average template

        Returns: True if registration is success

        """
        ref_flag = 1
        if ref_path == 'T1':
            ref_flag = 0

        for directory in os.walk(in_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)

                if re.search(regex, input_file):
                    output_file = out_folder+str(input_file.rsplit('/',1)[1])
                    output_path = output_file.split('.nii.gz')[0]
                    output_file = output_path + suffix
                    if ref_flag == 0:
                        ref_path = input_file.split('-DTI_MD.nii.gz')[0]
                        ref_path += '-T1_brain.nii.gz'
                    if not os.path.exists(output_file):
                        print("Aligning " + input_file +
                              "\n with " + ref_path)
                        # Linear registration of brain with the template
                        subprocess.call('flirt -in {0} -ref {1} -out {2} '
                                        '-cost {3} -searchcost {4} -v'
                                        .format(input_file,
                                                ref_path,
                                                output_file,
                                                self.registration['cost'],
                                                self.registration[
                                                    'searchcost']),
                                        shell=True)
                        print('Generated aligned file: ' +
                                         output_file)
                    else:
                        print("File Already exists: " + output_file)
                    sys.stdout.flush()
        return True
    def average_template(self, regex, in_folder, out_folder, avg_filename):

        # Prepare the command for generating the average template
        avg_template = out_folder + avg_filename
        output_files = []
        ctr = 0
        for directory in os.walk(in_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)

                if re.search(regex, input_file):
                    #print(input_file, len(output_files))
                    output_files.append(input_file)
                    ctr += 1
        #print(ctr, len(output_files))
        if not os.path.exists(avg_template):
            command = 'fslmaths '
            command += ' -add '.join(output_files)
            command += ' -div ' + str(len(output_files)) + \
                       ' ' + avg_template
            print("Number of Images: " + str(len(output_files)))
            subprocess.call(command, shell=True)
            print("Study specific average template: " + avg_template)
        return True

    def linear_registration(self, ref_path, iteration):
        """
        This function registers each image in 'input_folder' to an average
        study template.
        Initially, any specific subject can be chosen as reference image to
        align all MR Images. The output aligned images are averaged to
        generate a study specific template to which all the MR Images are
        again re-registered. This removes the bias in selecting a specific
        subject as reference image.


        Args:
            ref_path: Reference subject MR Image file path for initial
                      registration
            iteration: Number of iterations for re-registration on the
            generated average template

        Returns: True if registration is success

        """
        # Store output file name for generating the average template
        output_files = []
        regex = r"-T1_brain\.nii\.gz$"
        #regex = r"-DTI_MO\.nii\.gz$"
        #reference = "subject"
        #reference = "mni"
        reference = "subject"
        split_on = ".nii.gz"

        # This function is called again to re-register on the average template
        if iteration == 0:
            reference = "avg_template"
            #reference = "mni"
        for directory in os.walk(self.input_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)

                if re.search(regex, input_file):
                    output_file = '{0}_{1}_aligned.nii.gz'. \
                        format(input_file.split(split_on)[0], reference)
                    output_matrix = '{0}_{1}_aligned_matrix.mat'. \
                        format(input_file.split(split_on)[0], reference)
                    if not os.path.exists(output_file):
                        print("Aligning " + input_file +
                              "\n with " + ref_path)
                        # Linear registration of brain with the template
                        subprocess.call('flirt -in {0} -ref {1} -out {2} '
                                        '-omat {3} -cost {4} -searchcost {5} -v'
                                        .format(input_file,
                                                ref_path,
                                                output_file,
                                                output_matrix,
                                                self.registration['cost'],
                                                self.registration[
                                                    'searchcost']),
                                        shell=True)
                        print('Generated aligned file: ' +
                                         output_file)
                        output_files.append(output_file)
                    else:
                        print("File Already exists: " + output_file)
        # If this is not the last iteration
        if iteration >= 1:
            # Prepare the command for generating the average template
            avg_template = '/'.join(ref_path.split('/')[:-1])
            avg_template += '/T1_avg_study_template.nii.gz'

            if not os.path.exists(avg_template):
                command = 'fslmaths '
                command += ' -add '.join(output_files)
                command += ' -div ' + str(len(output_files)) + \
                           ' ' + avg_template
                print("Number of Images: " + str(len(output_files)))
                subprocess.call(command, shell=True)
                print("Study specific average template: " + avg_template)

                # Re-register all MR Images on the average template
                #if self.linear_registration(avg_template, iteration - 1):
                #    print("Aligned images to average template")
        return True

    def move(self, regex, out):
        ctr = 0
        subprocess.call('mkdir {0}'
                        .format(out), shell=True)
        for directory in os.walk(self.input_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)
                if re.search(regex, input_file):
                    """
                    folder = '/'.join(input_file.split('/')[:6]) #Until 'Data'
                    folder += '/FAMaps/'
                    folder += '/'.join(input_file.split('/')[7:])

                    folder = folder.rsplit('/',1)[0]
                    print("Creating folder "+folder)
                    """
                    output_file = out+'/'+str(input_file.rsplit('/',1)[1])
                    subprocess.call("cp {0} {1}".format(input_file,
                                                        output_file),
                                    shell=True)
                    print("File No:" +str(ctr+1)+" Generated "+output_file)
                    ctr += 1

    def gaussian_smoothing(self, regex, split_on, in_folder, out_folder):
        """
        This function uses gaussian function with given parameter sigma to
        smoothen all the aligned MR Images.

        """

        counter = 0

        for directory in os.walk(in_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)
                if re.search(regex, input_file):
                    output_file = out_folder + str(
                                        input_file.rsplit('/', 1)[1])
                    output_path = output_file.split(split_on)[0]
                    output_file = output_path + '_smoothed.nii.gz'

                    if not os.path.exists(output_file):
                        print("Smoothing image: " + input_file)
                        subprocess.call('fslmaths {0} -s {1} {2}'
                                        .format(input_file,
                                                self.registration[
                                                    'gauss_smooth_sigma'],
                                                output_file),
                                        shell=True)
                        counter += 1
                        print('File No.: ' + str(counter) +
                              ' Generated output: ' + output_file)
                sys.stdout.flush()

    def subsample(self, regex, split_on, in_folder, out_folder):
        """
        Given the gaussian smoothed images this function reduces
        the dimensionality of the images by half.
        """

        counter = 0

        for directory in os.walk(in_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)
                if re.search(regex, input_file):
                    output_file = out_folder + str(
                        input_file.rsplit('/', 1)[1])
                    output_path = output_file.split(split_on)[0]
                    output_file = output_path + '_subsampled.nii.gz'

                    if not os.path.exists(output_file):
                        print("Subsampling image: " + input_file)
                        subprocess.call('fslmaths {0} -subsamp2 {1}'
                                        .format(input_file,
                                                output_file),
                                        shell=True)
                        counter += 1
                        print('File No.: ' + str(counter) +
                              ' Generated output: ' + output_file)
                sys.stdout.flush()

    def rot_trans(self, regex, split_on, in_folder, out_folder):
        counter = 0
        for directory in os.walk(in_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)
                if re.search(regex, input_file):
                    for direction in ['x', 'y', 'z']:
                        x_axis = y_axis = z_axis = 0
                        if direction == 'x':
                            x_axis = 1
                        if direction == 'y':
                            y_axis = 1
                        if direction == 'z':
                            z_axis = 1
                        output_file = out_folder + str(
                                        input_file.rsplit('/', 1)[1])
                        output_path = output_file.split(split_on)[0]
                        output_file = output_path + \
                                        '_sub_rot3_{0}.nii.gz'.format(
                                            direction)

                        rot_matrix = 'rot3_{0}.mat'.format(direction)
                        angle_rot = 3 if random.uniform(0, 1) > 0.5 \
                            else -3
                        print("Rotating image: " + input_file)
                        subprocess.call('makerot -c {0} -a {1},{2},'
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
                        print("Generated image: "+output_file)

                        # Translating input image
                        mri_image = nb.load(input_file)
                        aff = mri_image.get_affine()
                        mri_image = mri_image.get_data()
                        translated_image = 0

                        # Translating rotated image
                        mri_image_rot = nb.load(output_file)
                        aff_rot = mri_image_rot.get_affine()
                        mri_image_rot = mri_image_rot.get_data()
                        translated_image_rot = 0

                        if x_axis == 1:
                            print("Translating in x-axis")
                            translated_image = shift(mri_image, [3, 0, 0],
                                                         mode='nearest')
                            translated_image_rot = shift(mri_image_rot, [3, 0,
                                                                       0],
                                                     mode='nearest')
                        if y_axis == 1:
                            print("Translating in y-axis")
                            translated_image = shift(mri_image, [0, 3, 0],
                                                     mode='nearest')
                            translated_image_rot = shift(mri_image_rot, [0, 3,
                                                                       0],
                                                     mode='nearest')
                        if z_axis == 1:
                            print("Translating in z-axis")
                            translated_image = shift(mri_image, [0, 0, 3],
                                                     mode='nearest')
                            translated_image_rot = shift(mri_image_rot, [0, 0,
                                                                       3],
                                                     mode='nearest')
                        im = nb.Nifti1Image(translated_image, affine=aff)
                        output_file = output_path + \
                                      '_sub_trans3_{0}.nii.gz'.format(
                                          direction)
                        nb.save(im, output_file)
                        print("Saving to " + output_file)

                        im_rot = nb.Nifti1Image(translated_image_rot,
                                               affine=aff_rot)
                        output_file = output_path + \
                                      '_sub_rot3_trans3_{0}.nii.gz'.format(
                                          direction)
                        nb.save(im_rot, output_file)
                        print("Saving to " + output_file)

                        counter += 1
                        print('File No.: ' + str(counter) +
                              ' Generated output: ' + output_file)

    def rotate(self, regex, split_on, in_folder, out_folder):
        """
        Given input images, this function augments the dataset
        by adding varying rotations to the MR Images.
        """

        counter = 0

        for directory in os.walk(in_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)
                if re.search(regex, input_file):
                    for direction in ['x', 'y', 'z']:
                        x_axis = y_axis = z_axis = 0
                        if direction == 'x':
                            x_axis = 1
                        if direction == 'y':
                            y_axis = 1
                        if direction == 'z':
                            z_axis = 1

                        output_file = out_folder + str(
                            input_file.rsplit('/', 1)[1])
                        output_path = output_file.split(split_on)[0]
                        output_file = output_path + \
                                      '_sub_rot5_{0}.nii.gz'.format(direction)
                        rot_matrix = 'rot5_{0}.mat'.format(direction)
                        angle_rot = 5 if random.uniform(0, 1) > 0.5 \
                            else -5
                        print("Rotating image: " + input_file)
                        subprocess.call('makerot -c {0} -a {1},{2},'
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
                        counter += 1
                        print('File No.: ' + str(counter) +
                              ' Generated output: ' + output_file)

        return True
