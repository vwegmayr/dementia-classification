from os import path
import argparse
import sys

from dementia_prediction.config_wrapper import Config
from dementia_prediction.preprocessing.data_pipeline import DataPipeline

config = Config()

parser = argparse.ArgumentParser(description="Preprocess the T1 data")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()

config.parse(path.abspath(args.paramfile))
paths = config.config.get('paths')
data_path = path.abspath(paths['data'])

# Initialise the Data preprocessing pipeline
pipeline = DataPipeline(in_folder=data_path,
                        params=config.config.get('parameters'))

if pipeline.ASL_preprocess():
    print("ASL preprocessing.")

