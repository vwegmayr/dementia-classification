from os import path
import os
import pickle
import re
import argparse
import sys

from dementia_prediction.config_wrapper import Config
from dementia_prediction.perceptron.data_input_balanced import DataInput
from dementia_prediction.perceptron.model import MultiLayerPerceptron

config = Config()
param_file = sys.argv[1]
config.parse(path.abspath(param_file))


paths = config.config.get('data_paths')

patients_dict = pickle.load(open(path.abspath(paths['class_labels']), 'rb'))
valid_patients = pickle.load(open(path.abspath(paths['valid_path']), 'rb'))

T1_folder = paths['T1_path']
DTI_MO_folder = paths['DTI_MO_path']
DTI_MD_folder = paths['DTI_MD_path']
DTI_FA_folder = paths['DTI_FA_path']

pos_train_filenames = []
neg_train_filenames = []

t1_pos_valid_filenames = []
t1_neg_valid_filenames = []
cbf_pos_valid_filenames = []
cbf_neg_valid_filenames = []
dti_mo_pos_valid_filenames = []
dti_mo_neg_valid_filenames = []
dti_fa_pos_valid_filenames = []
dti_fa_neg_valid_filenames = []
dti_md_pos_valid_filenames = []
dti_md_neg_valid_filenames = []


for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"-" + "CBF_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file. \
                rsplit('-CBF_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict:
                # Adding other modalities of the patient
                # T1
                patient = input_file.rsplit('/', 1)[1]
                pat_code = patient.split('-CBF_subsampled.nii.gz')[0]
                T1_filename = T1_folder + pat_code + '/' + pat_code
                T1_filename += '-T1_brain_subsampled.nii.gz'

                # DTI_MO
                patient = input_file.rsplit('/', 1)[1]
                pat_code = patient.split('-CBF_subsampled.nii.gz')[0]
                DTI_MO_filename = DTI_MO_folder + '/' + pat_code
                DTI_MO_filename += '-DTI_MO_subsampled.nii.gz'

                # DTI_MD
                patient = input_file.rsplit('/', 1)[1]
                pat_code = patient.split('-CBF_subsampled.nii.gz')[0]
                DTI_MD_filename = DTI_MD_folder + '/' + pat_code
                DTI_MD_filename += '-DTI_MD_subsampled.nii.gz'

                # DTI_FA
                patient = input_file.rsplit('/', 1)[1]
                pat_code = patient.split('-CBF_subsampled.nii.gz')[0]
                DTI_FA_filename = DTI_FA_folder + pat_code + '/' + pat_code
                DTI_FA_filename += '-DTI_FA_subsampled.nii.gz'

                if patient_code not in valid_patients:
                    if patients_dict[patient_code] == 0:
                        pos_train_filenames.append(input_file)
                        pos_train_filenames.append(T1_filename)
                        pos_train_filenames.append(DTI_MO_filename)
                        pos_train_filenames.append(DTI_MD_filename)
                        pos_train_filenames.append(DTI_FA_filename)

                    if patients_dict[patient_code] == 1:
                        neg_train_filenames.append(input_file)
                        neg_train_filenames.append(T1_filename)
                        neg_train_filenames.append(DTI_MO_filename)
                        neg_train_filenames.append(DTI_MD_filename)
                        neg_train_filenames.append(DTI_FA_filename)
                if patient_code in valid_patients:
                    if patients_dict[patient_code] == 0:
                        cbf_pos_valid_filenames.append(input_file)
                        t1_pos_valid_filenames.append(T1_filename)
                        dti_mo_pos_valid_filenames.append(DTI_MO_filename)
                        dti_md_pos_valid_filenames.append(DTI_MD_filename)
                        dti_fa_pos_valid_filenames.append(DTI_FA_filename)

                    if patients_dict[patient_code] == 1:
                        cbf_neg_valid_filenames.append(input_file)
                        t1_neg_valid_filenames.append(T1_filename)
                        dti_mo_neg_valid_filenames.append(DTI_MO_filename)
                        dti_md_neg_valid_filenames.append(DTI_MD_filename)
                        dti_fa_neg_valid_filenames.append(DTI_FA_filename)


#print("Total Number of valid patients: "+str(len(neg_valid_filenames)+len(
# pos_valid_filenames)))
#print("Total Number of train patients: "+str(len(neg_train_filenames)+len(
# pos_train_filenames)))
print("Train pos patients: " + str(len(pos_train_filenames)))
print("Train neg patients: " + str(len(neg_train_filenames)))
#print("Valid pos patients: "+ str(len(pos_valid_filenames)))
#print("Valid neg patients: "+ str(len(neg_valid_filenames)))

train = (pos_train_filenames, neg_train_filenames)
validation_t1 = (t1_pos_valid_filenames, t1_neg_valid_filenames)
validation_dti_mo = (dti_mo_pos_valid_filenames, dti_mo_neg_valid_filenames)
validation_dti_md = (dti_md_pos_valid_filenames, dti_md_neg_valid_filenames)
validation_dti_fa = (dti_fa_pos_valid_filenames, dti_fa_neg_valid_filenames)
validation_cbf = (cbf_pos_valid_filenames, cbf_neg_valid_filenames)
#print("T1:", len(validation_t1[0]))
train_data = DataInput(params=config.config.get('parameters'),
                       data=train, name='train')
validation_data = [DataInput(params=config.config.get('parameters'),
                            data=validation_t1, name='t1_valid'),
                    DataInput(params=config.config.get('parameters'),
                                                data=validation_dti_mo,
                              name='dti_mo_valid'),
                    DataInput(params=config.config.get('parameters'),
                                                data=validation_dti_md,
                              name='dti_md_valid'),
                    DataInput(params=config.config.get('parameters'),
                                                data=validation_dti_fa,
                              name='dti_fa_valid'),
                    DataInput(params=config.config.get('parameters'),
                                                data=validation_cbf,
                              name='cbf_valid')
                   ]
mlp_model = MultiLayerPerceptron(params=config.config.get('parameters'))
mlp_model.train(train_data, validation_data, True)


