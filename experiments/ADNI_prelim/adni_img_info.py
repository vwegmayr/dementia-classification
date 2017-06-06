"""Plots a histogram of each column in current folder"""
import argparse
from dementia_prediction.preprocessing.spreadsheet import SpreadSheet

parser = argparse.ArgumentParser(description="Analysis of ADNI image info.")
parser.add_argument("file", type=str, help="Path to spreadsheet csv file.")
args = parser.parse_args()

adni_sheet = SpreadSheet(args.file)

for field in adni_sheet.fields:
    adni_sheet.plt_hist(field, bins=20)