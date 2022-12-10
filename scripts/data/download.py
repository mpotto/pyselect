import argparse

from utils import get_folder, simple_download_from_url, download_from_url

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d")
args = parser.parse_args()

DATASET = args.dataset

URL = {
    "cpusmall": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/",
    "yearprediction": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/",
    "ailerons": "https://www.dcc.fc.up.pt/~ltorgo/Regression/",
    "amazon": "https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews",
    "higgs": "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/",
}
FILES = {
    "cpusmall": ["cpusmall"],
    "ailerons": ["ailerons.tgz"],
    "amazon": [""],
    "yearprediction": ["YearPredictionMSD.bz2"],
    "higgs": ["HIGGS.csv.gz"],
}

output_folder = get_folder(f"data/raw/{DATASET}")

for file in FILES[DATASET]:

    file_url = f"{URL[DATASET]}/{file}"
    file_path = f"{output_folder}/{file}"

    if file == "":
        raise ValueError("Cannot be downloaded. Check repo for specific instructions.")

    try:
        download_from_url(file_url, file_path)
    except KeyError:
        simple_download_from_url(file_url, file_path)
