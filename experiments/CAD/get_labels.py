import csv
from collections import defaultdict
import pickle

filep = open('/home/rams/4_Sem/Thesis/Data/CADDementia/CAD_labels.csv', 'r')
csvreader = csv.reader(filep)

cad_dict = defaultdict(int)
for row in csvreader:
    if row[0] != 'Patient': 
	    if row[1] == 'Normal': # NC which corresponds to 0 in ADNI
		cad_dict[row[0]] = 0
	    elif row[1] == 'MCI': # MCI which corresponds to 1 in ADNI
		cad_dict[row[0]] = 1
	    elif row[1] == 'AD': # AD which corresponds to 2 in ADNI
		cad_dict[row[0]] = 2

with open('./cad_valid_dict.pkl', 'wb') as filep:
    pickle.dump(cad_dict, filep)

with open('./cad_valid_dict.pkl', 'rb') as filep:
    cad_dict = pickle.load(filep)

for key, value in cad_dict.items():
    print(key, value)
