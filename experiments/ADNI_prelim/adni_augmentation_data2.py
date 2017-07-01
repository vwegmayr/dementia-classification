
import csv
from collections import defaultdict
import pickle


filep = open('/home/rams/PolyBox/Thesis/ADNI/labels.csv')
csvreader = csv.reader(filep)
patient_dict = [[] for i in range(0, 3)]
for i in range(0,7):
    patient_dict[0].append([])
    patient_dict[1].append([])
    patient_dict[2].append([])

with open('/home/rams/4_Sem/Thesis/Data/ADNI/Viktor/ignore_list.pkl', 'rb') as filep:
    ignore_list = pickle.load(filep)
print(len(ignore_list))
classes = {'Normal': 0, 'SMC': 1, 'LMCI': 2, 'EMCI': 3, 'MCI': 4, 'AD': 5,
           '': 6}
scanner_type = {'Field Strength=1.5':0, 'Field Strength=3.0':1, 'Field Strength=2.9':2}
with open('./adni_all_data.pkl', 'rb') as filep:
    all_pat = pickle.load(filep)
all_ctr = 0
for row in csvreader:
    if row[0][0] != 'i': #Header
        label = classes[row[1]]
        patient_code = row[0]
        scanner = scanner_type[row[2]]
        print(scanner, patient_code, label)
        if patient_code in all_pat:
            patient_dict[scanner][label].append(patient_code)

valid_dict = {}
train_dict = {}
for i in range(0, 3):
    print('Scanner: ', i)
    for j in range(0, 7):
        print('Class:', j, "Count: ", len(patient_dict[i][j]))

# Validation data should be balanced across scanners 1.5T and 3T
#TODO: Only take from pure MCI for validation data
# NC validation data patient_dict[0/1][0]
nc_valid = patient_dict[0][0][:250] + patient_dict[1][0][:250]
valid_dict.update(zip(nc_valid, [0 for i in range(0, 500)]))
nc_for_aug = patient_dict[0][0][250:1544] + patient_dict[1][0][250:1648]
# MCI patient_dict[0/1][4] - 250 1.5T, 3 T : 100 EMCI, 100 LMCI, 50 MCI
mci_valid = patient_dict[0][4][:250] + patient_dict[1][2][:100] + \
            patient_dict[1][3][:100] + patient_dict[1][4][:50]
valid_dict.update(zip(mci_valid, [4 for i in range(0, 500)]))

# AD patient_dict[0/1][5] - 250 1.5T 250 3T
ad_valid = patient_dict[0][5][:250] + patient_dict[1][5][:250]
valid_dict.update(zip(ad_valid, [5 for i in range(0, 500)]))
ad_for_aug = patient_dict[0][5][250:] + patient_dict[1][5][250:]

mci_train = []
nc_train = []
ad_train = []
for patient, label in all_pat.items():
    if patient not in valid_dict:
        train_dict[patient] = label
        if label == 0 or label == 1:
            nc_train.append(patient)
        if label == 2 or label == 3 or label == 4:
            mci_train.append(patient)
        if label == 5:
            ad_train.append(patient)
print(len(nc_train), len(mci_train), len(ad_train))
print("NC:", len(nc_for_aug), "AD:", len(ad_for_aug))

with open('./adni_ad_aug_data2.pkl', 'wb') as filep:
    pickle.dump(ad_for_aug, filep)
with open('./adni_nc_aug_data2.pkl', 'wb') as filep:
    pickle.dump(nc_for_aug, filep)

with open('./adni_train_data2_aug.pkl', 'wb') as filep:
    pickle.dump(train_dict, filep)
with open('./adni_valid_data2_aug.pkl', 'wb') as filep:
    pickle.dump(valid_dict, filep)
