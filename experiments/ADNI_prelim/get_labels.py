# Small script to create training and validation dictionaries from ADNI
# spreadsheet data

import csv
from collections import defaultdict
import pickle


filep = open('/home/rams/PolyBox/ise-squad/ADNI/spreadsheet/labels.csv')
csvreader = csv.reader(filep)
patient_dict = defaultdict(int)
ad_ctr = 0
emci_ctr = 0
lmci_ctr = 0
mci_ctr = 0
smc_ctr = 0
nc_ctr = 0

with open('./ignore_list.pkl', 'rb') as filep:
    ignore_list = pickle.load(filep)
print(len(ignore_list))
for row in csvreader:
    if row[0][0] != 'i': #Header
        label = row[1]
        patient_code = row[0]
        if label == 'Normal':
            patient_dict[patient_code] = 0
            nc_ctr += 1
        elif label == 'SMC':
            patient_dict[patient_code] = 1
            smc_ctr += 1
        elif label == 'LMCI':
            patient_dict[patient_code] = 2
            lmci_ctr += 1
        elif label == 'EMCI':
            patient_dict[patient_code] = 3
            emci_ctr += 1
        elif label=='MCI':
            patient_dict[patient_code] = 4
            mci_ctr += 1
        elif label == 'AD':
            patient_dict[patient_code] = 5
            ad_ctr += 1
        else:
            print(row)

#print(list(patient_dict))
with open("adni_all_patients.pkl", 'wb') as filep:
    pickle.dump(patient_dict, filep)
valid_ad = 0
valid_mci = 0
valid_nc = 0
valid_patients = defaultdict(int)
training_patients = defaultdict(int)

for key, value in patient_dict.items():
    if int(key) not in ignore_list:
        if value == 0 and valid_nc < 500:
            valid_patients[key] = value
            valid_nc += 1
        elif value == 4 and valid_mci < 500:
            valid_patients[key] = value
            valid_mci += 1
        elif value == 5 and valid_ad < 500:
            valid_patients[key] = value
            valid_ad += 1
        else:
            training_patients[key] = value

print("AD:", ad_ctr, "MCI: ", mci_ctr, "LMCI:", lmci_ctr, "EMCI:", emci_ctr,
      "SMC: ", smc_ctr, "NC: ", nc_ctr)
print(valid_nc, valid_mci, valid_ad)
print("Valid: ", len(valid_patients))
with open("adni_valid_patients.pkl", 'wb') as filep:
    pickle.dump(valid_patients, filep)
with open("adni_train_patients.pkl", 'wb') as filep:
    pickle.dump(training_patients, filep)
print(len(training_patients))
