"""
Script to assign labels to the images from ADNI downloaded DTI data and 
split the
total data to training and validation data.
Input: ADNIMERGE.csv, ADNI subject filenames list, ADNI search subjectfile 
for baseline dx
"""

import csv
import datetime
from datetime import date
import argparse
import sys
import pickle

parser = argparse.ArgumentParser(description="Get labels for the images.")
parser.add_argument("mergefile", type=str, help='Path to the ADNIMERGE.csv '
                                                'file')
parser.add_argument("subjectfile", type=str, help='Path to the ADNI DTI '
                                                  'subjects filenames list')
# Of the format [subject_id_date, ....] no file extension in the name
parser.add_argument("searchfile", type=str, help='Path to the ADNI DTI '
                                                  'Advanced Search (Beta) csv')
args = parser.parse_args()

merge_reader = csv.reader(open(args.mergefile))
with open(args.subjectfile, 'rb') as filep:
    subject_reader = pickle.load(filep)

# Returns the label for the diagnosis given to the subject at a visit
def get_label(dx):
    if dx == '':
        return -1
    elif dx == 'NL' or dx == 'MCI to NL' or dx == 'Dementia to NL':
        return 0
    elif dx == 'MCI' or dx == 'NL to MCI' or dx == 'Dementia to MCI':
        return 1
    elif dx == 'Dementia' or dx == 'NL to Dementia' or dx == 'MCI to Dementia':
        return 2
    elif dx == 'Normal' or dx == 'SMC' or dx == 'CN':
        return 0
    elif dx == 'EMCI' or dx == 'MCI' or dx == 'LMCI':
        return 1
    elif dx == 'AD':
        return 2
    else:
        print("Not a proper diagnosis: "+dx+". Stop and recheck the file.")
        sys.exit(0)

# Get the screening labels for patients. Use it when patient not found in
# ADNIMERGE.csv
screening_dict = {}
with open(args.searchfile, 'r') as filep:
    search_reader = csv.reader(filep)
    # row[0] : subject_id # row[2] : dx_screening
    for row in search_reader:
        if row[0][0] != 'S':
            screening_dict[row[0]] = get_label(row[2])

print(len(screening_dict))

# Stores the diagnosis label available for each patient for all his visits
# in the format {"patient_id_1":{'visit_date':dx,...}, "patient_id_2":{....},.}
patient_visit_dx = {}

# Read "PTID", "EXAMDATE" and "DX" columns from ADNIMERGE.csv
for row in merge_reader:
    if row[0] != 'RID': # Ignoring header row
        patient_id = row[1]
        exam_date = row[6] # Date should be of ISO 8061 format
        label = get_label(row[51])
        if patient_id not in patient_visit_dx:
            patient_visit_dx[patient_id] = {}
        patient_visit_dx[patient_id][exam_date] = label

# Stores the label for each image downloaded from ADNI
image_label_dict = {}

# Stores the
unique_patients = []
# Read the subject filenames from the list of DTI filenames
for dti_file in subject_reader:
    split_file = dti_file.split('_')
    patient_id = split_file[0]+'_'+split_file[1]+'_'+split_file[2]
    unique_patients.append(patient_id)
    date_string = split_file[3][0:8]
    iso_date = '{}-{:02}-{:02}'.format(date_string[0:4], int(date_string[4:6]),
                                       int(date_string[6:8]))
    if patient_id in patient_visit_dx:
        visits = sorted(patient_visit_dx[patient_id]) # sorted list of visits
        # Initialize the label with the diagnosis of the baseline visit
        image_label_dict[dti_file] = patient_visit_dx[patient_id][visits[0]]
        # Check for any change of label for that study date
        #print(visits)
        for v in visits:
            if iso_date < v: #TODO: Check whether the ADNIMERGE visits
                # are after or before the subject file visits. Here,
                # it is assumed they are before.
                break
            elif patient_visit_dx[patient_id][v] != -1: # Don't change
                # if there is no dx available. Keep the previous diagnosis.
                image_label_dict[dti_file] = patient_visit_dx[patient_id][v]
        if image_label_dict[dti_file] == -1:
            print("No proper label assigned for patient: "+patient_id)
            image_label_dict[dti_file] = screening_dict[patient_id]
            print("Giving ", screening_dict[patient_id])

    else:
        print("Error: Patient "+patient_id+" not found in ADNIMERGE")
        image_label_dict[dti_file] = screening_dict[patient_id]
        print("Giving ", screening_dict[patient_id])

# Store the dictionary
with open('ADNI_DTI_labels.csv', 'w') as filep:
    label_writer = csv.writer(filep)
    for dti_file, label in image_label_dict.items():
        label_writer.writerow([dti_file, label])

with open('ADNI_DTI_all_dict.pkl', 'wb') as filep:
    pickle.dump(image_label_dict, filep)

print("Total images:", len(image_label_dict),
      "Total Patients:", len(set(unique_patients)))
