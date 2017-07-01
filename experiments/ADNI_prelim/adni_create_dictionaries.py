# Small script to create training and validation dictionaries from ADNI
# spreadsheet data

import csv
from collections import defaultdict
import pickle

filep = open('/home/rams/PolyBox/Thesis/ADNI/labels.csv')
csvreader = csv.reader(filep)

# Counting scanner types and class types for image ids
image_dict = [defaultdict(int) for i in range(0, 3)]

# Counting scanner types and class types for patient ids
patient_dict = [[] for i in range(0, 3)]
for i in range(0, 3):
    for j in range(0, 6):
         patient_dict[i].append([])
# Mapping imageId to patient ID
image_patient = defaultdict(int)

'''
with open('./ignore_list.pkl', 'rb') as filep:
    ignore_list = pickle.load(filep)
print(len(ignore_list))
'''
scanner_type = {'Field Strength=1.5':0, 'Field Strength=3.0':1, 'Field Strength=2.9':2}
classes = {'Normal': 0, 'SMC': 1, 'LMCI': 2, 'EMCI': 3, 'MCI': 4, 'AD': 5,
           '': 6}

with open('./adni_new_dict.pkl', 'rb') as filep:
    all_data = pickle.load(filep)
print("All:", len(all_data))
train_subjects = []
all_subjects = []
valid_subjects = []
patient_id_class = defaultdict(int)

for row in csvreader:
    if row[0][0] != 'S': #Header
        patient_id = row[0]
        #print(scanner, image_code, label)
        if patient_id not in patient_id_class:
            patient_id_class[patient_id] = row[2]

        elif patient_id_class[patient_id] != row[2]:
            print("Evaluation Changed")
print(len(patient_id_class))

for row in csvreader:
    if row[0][0] != 'i': #Header
        label = classes[row[1]]
        image_code = row[0]
        scanner = scanner_type[row[2]]
        patient_id = row[3]
        #print(scanner, image_code, label)
        if image_code in all_data:
            image_dict[scanner][label] += 1
            if patient_id not in patient_id_class:
                patient_id_class[patient_id] = label
            elif patient_id_class[patient_id] != label:
                print("Evaluation Changed")
        if image_code in all_data and patient_id not in all_subjects:
            patient_dict[scanner][label].append(patient_id)
        all_subjects.append(patient_id)
        image_patient[image_code] = patient_id
        '''
        if image_code in train:
            train_subjects.append(patient_id)
        if image_code in valid:
            valid_subjects.append(patient_id)
        '''
print('Unique Patients in Total:', len(patient_id_class))
'''
# Set of unique train,valid patients
train_set = set(train_subjects)
valid_set = set(valid_subjects)
all_set = set(train_subjects+valid_subjects)
print(len(train_set), len(valid_set), len(all_set), len(all_set)-len(train_set))

corrected_valid_dict_duplicates = {}
corrected_valid_dict = {}
for key, value in valid.items():
    if image_patient[key] not in train_subjects:
        corrected_valid_dict_duplicates[key] = value
        corrected_valid_dict[key] = image_patient[key]
#print(corrected_valid_dict_duplicates)
#print(corrected_valid_dict)
print(len(train_subjects), len(valid_subjects), len(all_subjects))
'''
valid_dict_new = []
for scanner_name, type in scanner_type.items():
    print("Scanner:", scanner_name)
    for j in range(0, 6):
        print(j, len(patient_dict[type][j]))
valid_dict = {}

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
