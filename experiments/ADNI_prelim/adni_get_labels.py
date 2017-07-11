"""
Script to assign labels to the images from ADNI downloaded data and split the
total data to training and validation data.
Input: ADNIMERGE.csv, ADNI 'Advanced Search (Beta)' subject info csv file.
Output: A split of training and validation dictionaries for the ADNI data.
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
parser.add_argument("subjectfile", type=str, help='Path to the ADNI Advanced '
                                    'Search (Beta) subject info spreadsheet')
args = parser.parse_args()

merge_reader = csv.reader(open(args.mergefile))
subject_reader = csv.reader(open(args.subjectfile))

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

# Stores the image_id and patient_id
image_patient = {}
# Read the "img_ID", "Subject ID" and "Study Date" from subjectfile
for row in subject_reader:
    if row[0] != 'img_ID': # Ignoring header row
        image_id = row[0]
        patient_id = row[24]
        image_patient[image_id] = patient_id
        date_string = row[23].split('/') # If date is of MM/DD/YYYY format.
        iso_date = '{}-{:02}-{:02}'.format(date_string[2], int(date_string[0]),
                                           int(date_string[1]))
        #print(iso_date)
        if patient_id in patient_visit_dx:
            visits = sorted(patient_visit_dx[patient_id]) # sorted list of visits
            # Initialize the label with the diagnosis of the baseline visit
            image_label_dict[image_id] = patient_visit_dx[patient_id][visits[0]]
            # Check for any change of label for that study date
            #print(visits)
            for v in visits:
                if iso_date < v: #TODO: Check whether the ADNIMERGE visits
                    # are after or before the subject file visits. Here,
                    # it is assumed they are before.
                    break
                elif patient_visit_dx[patient_id][v] != -1: # Don't change
                    # if there is no dx available. Keep the previous diagnosis.
                    image_label_dict[image_id] = patient_visit_dx[patient_id][v]
            if image_label_dict[image_id] == -1:
                print("No proper label assigned for patient: "+patient_id+" "
                    "image: "+image_id,"Assigning: ", row[1])
                # In this case, assign the DX_GROUP label. (Baseline label)
                image_label_dict[image_id] = get_label(row[1])

        else:
            print("Error: Patient "+patient_id+" not found in ADNIMERGE",
                  "Using baseline label: ", row[1])
            # In this case, use the DXGROUP variable of the subjectfile
            image_label_dict[image_id] = get_label(row[1])

# Store the dictionary
with open('ADNI_labels.csv', 'w') as filep:
    label_writer = csv.writer(filep)
    for image_id, label in image_label_dict.items():
        label_writer.writerow([image_id, label])

with open('ADNI_all_dict.pkl', 'wb') as filep:
    pickle.dump(image_label_dict, filep)

# Stub to compare the changes of labels during visits or conversions
with open('adni_all_data.pkl', 'rb') as filep:
    old_dict = pickle.load(filep)

changes_ctr = 0
conv_patients = []
with open('ADNI_conversions.csv','w') as filep:
    conv_writer = csv.writer(filep)
    conv_writer.writerow(['Img_ID', 'PTID', 'OldLabel', 'NewLabel'])
    for key, value in old_dict.items():
        old_label = 0
        if value == 0 or value == 1:
            old_label = 0
        elif value in [2,3,4]:
            old_label = 1
        elif value in [5]:
            old_label = 2
        if image_label_dict[key] != old_label:
            print("Diagnosis label changed from ", old_label, "to new ",
                  image_label_dict[key], "image: ", key)
            changes_ctr += 1
            conv_patients.append(image_patient[key])
            conv_writer.writerow([key, image_patient[key], old_label,
                                  image_label_dict[key]])
print("Total number of changes: ", changes_ctr)
print("Total number of patients converted/reverted:", len(set(conv_patients)))
# Verify with example S_915

# Ignoring some images, create new dictionary for the downloaded data
ADNI_new_dict = {}
with open('ADNI_new_dict.pkl', 'wb') as filep:
    for key, value in old_dict.items():
        ADNI_new_dict[key] = image_label_dict[key]
    pickle.dump(ADNI_new_dict, filep)