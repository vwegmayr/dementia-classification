"""
This script would create the training and validation data dictionaries.
Input: adni_all_data.pkl - Dictionary of all images and its labels
Output: adni_train.pkl adni_valid.pkl
"""
import csv
from collections import defaultdict
import pickle
import argparse
import sys

parser = argparse.ArgumentParser(description="Get labels for the images.")
parser.add_argument("dtialldict", type=str, help='Path to DTI All data dict')

args = parser.parse_args()

def get_distribution(all_data):
    # Counting scanner types and class types for patient ids
    patient_dict = [[] for i in range(0, 3)]
    for i in range(0,3):
        patient_dict[i].append([]) # For dtifilenames
        patient_dict[i].append([]) # For subject names

    # Mapping imageId to patient ID
    image_patient = defaultdict(int)


    train_subjects = []
    all_subjects = []
    valid_subjects = []

    for dti_file, label in all_data.items():
        patient_id = dti_file.split('_')[0] + '_' + \
                     dti_file.split('_')[1] + '_' + \
                     dti_file.split('_')[2]
        patient_dict[label][0].append(dti_file)
        patient_dict[label][1].append(patient_id)
        all_subjects.append(patient_id)

    unique_patients = list(set(all_subjects))
    print("Unique patients:", len(unique_patients))
    for i in range(0, 3):
        print("Class: ", i)
        print("Images: ", len(set(patient_dict[i][0])),
              "Unique patients: ", len(set(patient_dict[i][1])))
    return patient_dict, unique_patients

with open(args.dtialldict, 'rb') as filep:
    all_data = pickle.load(filep)
print("Length of Dictionary:", len(all_data))
patient_dict, unique_patients = get_distribution(all_data)
valid_patients = list(set(patient_dict[0][1]))[:10]  +\
                 list(set(patient_dict[1][1]))[:12] +\
                 list(set(patient_dict[2][1]))[:8]
valid_patients = list(set(valid_patients))
print(valid_patients)
print("Number of unique valid patients", len(valid_patients))

train_dict = {}
valid_dict = {}

for dti_file, label in all_data.items():
    patient_id = dti_file.split('_')[0]+'_'+\
              dti_file.split('_')[1] + '_'+\
              dti_file.split('_')[2]
    if patient_id not in valid_patients:
        train_dict[dti_file] = label
    else:
        valid_dict[dti_file] = label
print("Train dictionary length:", len(train_dict),
      "Valid dictionary length:", len(valid_dict))

patient_dict, _ = get_distribution(train_dict)
patient_dict, _ = get_distribution(valid_dict)

with open('ADNI_train_dti.pkl', 'wb') as filep:
    pickle.dump(train_dict, filep)

with open('ADNI_valid_dti.pkl', 'wb') as filep:
    pickle.dump(valid_dict, filep)
