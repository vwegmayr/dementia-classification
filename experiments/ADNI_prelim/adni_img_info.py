"""Plots a histogram of each column in current folder"""
import argparse
from dementia_prediction.preprocessing.spreadsheet import SpreadSheet

parser = argparse.ArgumentParser(description="Analysis of ADNI image info.")
parser.add_argument("file", type=str, help="Path to spreadsheet csv file.")
parser.add_argument(
    "output",
    choices=["hist","bar_unique","fields","bar_stack"],
    nargs="+",
    help="List which outputs you want to produce."
    )
parser.add_argument("--fields", nargs="*")
args = parser.parse_args()



adni_sheet = SpreadSheet(args.file)

if "hist" in args.output:
    for field in adni_sheet.fields:
        adni_sheet.plt_hist(field, bins=20)

if "bar_unique" in args.output:
    for field in adni_sheet.fields:
        adni_sheet.plt_bar_unique(field)

if "fields" in args.output:
    print(adni_sheet.fields)

if "bar_stack" in args.output:
    if args.fields != None and len(args.fields) == 2:
        adni_sheet.plt_bar_stack(args.fields[0], args.fields[1])
    else:
        parser.error("You need to specify two fields for bar_stack.")

