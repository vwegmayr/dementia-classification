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
parser.add_argument("t2alldict", type=str, help='Path to T2 All data dict')
parser.add_argument("t2searchfile", type=str, help='Path to T2 Advanced '
                                                   'Search (Beta) file')

args = parser.parse_args()


# Counting scanner types and class types for patient ids
patient_dict = [[] for i in range(0, 3)]
for i in range(0, 3):
    for j in range(0, 3):
         patient_dict[i].append([])
         patient_dict[i][j].append([]) # For subjects
         patient_dict[i][j].append([])  # For image ids

# Mapping imageId to patient ID
image_patient = defaultdict(int)

scanner_type = {'Field Strength=1.5':0,
                'Field Strength=3.0':1,
                'Field Strength=2.9':2
                }

with open(args.t2alldict, 'rb') as filep:
    all_data = pickle.load(filep)
print("All:", len(all_data))

csvreader = csv.reader(open(args.t2searchfile, 'r'))

train_subjects = []
all_subjects = []
valid_subjects = []

for row in csvreader:
    if row[0][0] != 'S': #Header
        image_code = 'I'+row[7]
        if image_code in all_data:
            label = all_data[image_code]
            scanner = scanner_type[row[6]]
            patient_id = row[0]
            patient_dict[label][scanner][0].append(patient_id)
            patient_dict[label][scanner][1].append(image_code)
            image_patient[image_code] = patient_id
            all_subjects.append(patient_id)
unique_patients = list(set(all_subjects))
print(len(unique_patients))
for i in range(0, 3):
    print("Class: ", i)
    for j in range(0, 3):
        print("Scanner: ", j, "Images: ", len(set(patient_dict[i][j][1])),
              "Unique patients: ", len(set(patient_dict[i][j][0])))
    print("Total:", len(set(patient_dict[i][0][1])) +len(set(patient_dict[i][
                                                                1][1]))
          + len(set(patient_dict[i][2][1])) )
class_0 = list(set(patient_dict[0][0][0] + patient_dict[0][1][0] + \
                patient_dict[0][2][0]))
print(len(class_0))
class_1 = list(set(patient_dict[1][0][0] + patient_dict[1][1][0] + \
                patient_dict[1][2][0]))
print(len(class_1))
class_2 = list(set(patient_dict[2][0][0] + patient_dict[2][1][0] + \
                patient_dict[2][2][0]))
print(len(class_2))

# NC , MCI conversions/reversions
print("NC<->MCI:", len(class_0)+len(class_1) - len(set(class_0 + class_1)))
# MCI, AD conversions/reversions
print("MCI<->AD:", len(class_1)+len(class_2) - len(set(class_1 + class_2)))
# NC , AD conversions/reversions
print("NC<->AD:", len(class_0)+len(class_2) - len(set(class_0 + class_2)))

valid_subjects = list(set(class_0[:15] + class_1[:15] + class_2[:10]))
unique_patients = list(set(all_subjects))
for subject in unique_patients:
    if subject not in valid_subjects:
        train_subjects.append(subject)
print(len(train_subjects)+len(valid_subjects), len(train_subjects), len(valid_subjects))


valid_dict = {}
train_dict = {}
for img_code, label in all_data.items():
    if image_patient[img_code] in train_subjects:
        train_dict[img_code] = label
    elif image_patient[img_code] in valid_subjects:
        valid_dict[img_code] = label

'''
with open('ADNI_train_t2.pkl', 'wb') as filep:
    pickle.dump(train_dict, filep)

with open('ADNI_valid_t2.pkl', 'wb') as filep:
    pickle.dump(valid_dict, filep)
'''
'''
# NC validation data patient_dict[0/1][0]
nc_valid = patient_dict[0][0][:20] + patient_dict[1][0][:20]
valid_dict.update(zip(nc_valid, [0 for i in range(0, len(nc_valid))]))

# MCI patient_dict[0/1][4] - 250 1.5T, 3 T : 100 EMCI, 100 LMCI, 50 MCI
mci_valid = patient_dict[0][4][:40]
valid_dict.update(zip(mci_valid, [4 for i in range(0, len(mci_valid))]))
print(len(mci_valid))

# AD patient_dict[0/1][5] - 250 1.5T 250 3T
ad_valid = patient_dict[0][5][:20] + patient_dict[1][5][:20]
valid_dict.update(zip(ad_valid, [5 for i in range(0, len(ad_valid))]))
print("Validation Data:", len(valid_dict), "NC:", len(nc_valid), "MCI:",
      len(mci_valid), "AD:", len(ad_valid))

train_dict_new = {}
valid_dict_new = {}
# If the image id corresponding patient id in valid_dict, create new valid dict
# with that image id
patient_ctr = defaultdict(int)
for key, value in all_data.items():
    if image_patient[key] in valid_dict:
        valid_dict_new[key] = value
    elif value != 1 or value != 2 or value != 3:
        train_dict_new[key] = value
        patient_ctr[image_patient[key]] += 1
print("Unique train patients:", len(patient_ctr))
print("Valid Dict:", len(valid_dict_new))
print("Train Dict:", len(train_dict_new))

with open('adni_valid_dict_data1.pkl', 'wb') as filep:
    pickle.dump(valid_dict_new, filep)

with open('adni_train_dict_data1.pkl', 'wb') as filep:
    pickle.dump(train_dict_new, filep)
'''