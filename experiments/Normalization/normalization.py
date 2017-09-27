from os import path
import pickle
import argparse
import re

from pathos.multiprocessing import ProcessPool
from dementia_prediction.config_wrapper import Config
from dementia_prediction.normalize import Normalize

config = Config()
parser = argparse.ArgumentParser(description="Normalize data")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
params = config.config.get('parameters')
print("Params:", params, flush=True)

valid_dict = pickle.load(open(params['valid_path'], 'rb'))
print("Valid", len(valid_dict), "Patients")

train_dict = pickle.load(open(params['train_path'], 'rb'))
print("Train", len(train_dict), "Patients")

norm_object = Normalize(params, train_dict, valid_dict)
print(params['regex'])
train_patients, valid_patients = norm_object.get_files(params['smooth_path'],
                                                       regex=r""+params['regex']+"$",
                                                       split_on=params[
                                                           'split_on']
                                                       )
print("Train: ", len(train_patients), "Valid:", len(valid_patients))
num_parallel = 15
pool = ProcessPool(num_parallel)
all_patients = train_patients + valid_patients # per-image normalization of all images
split = int(len(all_patients) / num_parallel)
splits = []
for par in range(0, num_parallel - 1):
    splits.append(all_patients[par * split:(par + 1) * split])
splits.append(all_patients[(num_parallel - 1) * split:])

print("Finding per-image normalization..")
pool.map(norm_object.per_image, splits)
print("Retrieving files from ", params['per_image_out'])
train_patients, valid_patients = norm_object.get_files(params['per_image_out'],
                                                       regex=r""+params['regex']+"$",
                                                       split_on=params[
                                                           'split_on']
                                                       )

all_patients = train_patients + valid_patients
print("All patients:", len(all_patients), "Train:", len(train_patients), "Valid:", len(valid_patients), flush=True)
if params['only_test'] == 'True':
    mean_path  = params['norm_mean_var'] + params['mode'] + '_data_mean.pkl'
    with open(mean_path, 'rb') as filep:
        norm_object.mean_norm = pickle.load(filep)
    var_path = params['norm_mean_var'] + params['mode'] + '_data_var.pkl'
    with open(var_path, 'rb') as filep:
        norm_object.var_norm = pickle.load(filep)
    print("mean:", len(norm_object.mean_norm), "Var:", len(norm_object.var_norm))
else:
    print("Finding mean, var normalization of ", len(train_patients), "images", flush=True)
    norm_object.mean_norm, norm_object.var_norm = norm_object.normalize(
                                                    train_patients)

#num_parallel = 10
#pool = ProcessPool(num_parallel)
# Applying mean, variance normalization
split = int(len(all_patients) /  num_parallel)
splits = []
for par in range(0, num_parallel - 1):
    splits.append(all_patients[par * split:(par + 1) * split])
splits.append(all_patients[(num_parallel - 1) * split:])
pool.map(norm_object.store, splits)
