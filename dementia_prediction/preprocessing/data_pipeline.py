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

    def brain_extraction(self):
        """
         This function extracts the brain from the raw T1 weighted MR Images
         using bet tool of FSL 5.0.
         It might take 5-6 minutes for an image of ~7MB.
         Run this function parallelly with different file paths to get
         optimal performance.
        """
        for directory in os.walk(self.input_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with "-T1.nii.gz"
                input_file = os.path.join(directory[0], file)

                if re.search(r'-T1\.nii\.gz$', input_file):
                    output_file = '{0}_brain.nii.gz'. \
                        format(input_file.split('.nii.gz')[0])

                    if not os.path.exists(output_file):
                        print("Extracting brain from file: " + input_file)
                        subprocess.call('bet {0} {1} -B -f {2} -v'
                                        .format(input_file,
                                                output_file,
                                                self.bet["frac_intens_thres"]),
                                        shell=True)
                        print('Generated extracted brain: ' + output_file)
                    else:
                        print("File already exists: " + output_file)
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
        reference = "subject"
        split_on = ".nii.gz"

        # This function is called again to re-register on the average template
        if iteration == 0:
            reference = "avg_template"

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
                                        '-omat {3} -cost {4} -searchcost {5}'
                                        .format(input_file,
                                                ref_path,
                                                output_file,
                                                output_matrix,
                                                self.registration['cost'],
                                                self.registration[
                                                    'searchcost']),
                                        shell=True)
                        print('Generated aligned file: ' + output_file)
                        output_files.append(output_file)
                    else:
                        print("File Already exists: " + output_file)

        # If this is not the last iteration
        if iteration >= 1:
            # Prepare the command for generating the average template
            avg_template = '/'.join(ref_path.split('/')[:-1])
            avg_template += '/avg_study_template.nii.gz'

            if not os.path.exists(avg_template):
                command = 'fslmaths '
                command += ' -add '.join(output_files)
                command += ' -div ' + str(len(output_files)) + \
                           ' ' + avg_template
                print("Number of Images: "+str(len(output_files)))
                subprocess.call(command, shell=True)
                print("Study specific average template: " + avg_template)

                # Re-register all MR Images on the average template
                if self.linear_registration(avg_template, iteration - 1):
                    print("Aligned images to average template")
        return True

    def gaussian_smoothing(self):
        """
        This function uses gaussian function with given parameter sigma to
        smoothen all the aligned MR Images.

        """
        regex = r"-T1_brain_avg_template_aligned\.nii\.gz$"
        split_on = "_avg_template_aligned.nii.gz"
        counter = 0
        
        for directory in os.walk(self.input_folder):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)
                if re.search(regex, input_file):
                    output_file = '{0}_smoothed.nii.gz'. \
                        format(input_file.split(split_on)[0])

                    if not os.path.exists(output_file):
                        print("Smoothing image: " + input_file)
                        subprocess.call('fslmaths {0} -s {1} {2}'
                                        .format(input_file,
                                                self.registration[
                                                    'gauss_smooth_sigma'],
                                                output_file),
                                        shell=True)
                        counter += 1
                        print('File No.: '+str(counter) +
                              ' Generated output: ' + output_file)

        return True
