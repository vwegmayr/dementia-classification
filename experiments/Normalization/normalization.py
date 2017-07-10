from os import path
import pickle
import argparse

from pathos.multiprocessing import ProcessPool
from dementia_prediction.config_wrapper import Config
from dementia_prediction.normalize import Normalize

config = Config()
parser = argparse.ArgumentParser(description="Normalize data")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
params = config.config.get('parameters')


valid_dict = pickle.load(open(params['valid_path'], 'rb'))
print("Valid", len(valid_dict), "Patients")

train_dict = pickle.load(open(params['train_path'], 'rb'))
print("Train", len(train_dict), "Patients")

norm_object = Normalize(params, train_dict, valid_dict)
train_patients, valid_patients = norm_object.get_files(params['smooth_path'],
                                                       regex=params['regex'],
                                                       split_on=params[
                                                           'split_on']
                                                       )
print("Train: ", len(train_patients), "Valid:", len(valid_patients))

num_parallel = 20
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
train_patients, valid_patients = norm_object.get_files(params['per_image_out'])
all_patients = train_patients + valid_patients

print("Finding mean, var normalization of ", len(train_patients), "images")
norm_object.mean_norm, norm_object.var_norm = norm_object.normalize(
                                                train_patients)

# Applying mean, variance normalization
split = int(len(all_patients) /  num_parallel)
splits = []
for par in range(0, num_parallel - 1):
    splits.append(all_patients[par * split:(par + 1) * split])
splits.append(all_patients[(num_parallel - 1) * split:])
pool.map(norm_object.store, splits)
