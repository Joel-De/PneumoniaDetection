import argparse
import json
import os

import cv2
import pandas as pd
import pydicom as dicom
from tqdm import tqdm


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", help="Path to common parent directory of dataset", required=True)
    p.add_argument("--output_dir", help="Where to save exported dataset to", required=True)
    args = p.parse_args()
    return args


if __name__ == '__main__':
    args = parseArgs()
    image_path = os.path.join(args.data_dir, "stage_2_train_images")
    labels = os.path.join(args.data_dir, "stage_2_train_labels.csv")
    detailed_info = os.path.join(args.data_dir, "stage_2_detailed_class_info.csv")


    trainDir = os.path.join(args.output_dir, "Train")
    testDir = os.path.join(args.output_dir, "Test")

    if not os.path.exists(trainDir):
        os.makedirs(os.path.join(trainDir, "Images"))

    if not os.path.exists(testDir):
        os.makedirs(os.path.join(testDir, "Images"))

    import random

    random.seed(42)
    dataset = {}
    data = pd.read_csv(labels)
    for item in data.iterrows():
        dataset[item[1]['patientId']] = {"Label": item[1]['Target']}

    data = pd.read_csv(detailed_info)
    for item in data.iterrows():
        dataset[item[1]['patientId']]["AdditionalInformation"] = item[1]['class']

    for file in os.listdir(image_path):
        patientID = os.path.splitext(file)[0]
        dataset[patientID]["FilePath"] = os.path.join(image_path, file)
        dataset[patientID]["patientID"] = patientID

    dataPoints = list(dataset.values())
    random.shuffle(dataPoints)

    # Generate Train & Test splits
    PercentTrain = 0.9

    trainSplit, testSplit = dataPoints[:int(len(dataPoints) * PercentTrain)], dataPoints[
                                                                              int(len(dataPoints) * PercentTrain):]
    for split in [(trainSplit, trainDir), (testSplit, testDir)]:
        datasetJson = []
        for data in tqdm(split[0]):
            ds = dicom.dcmread(data['FilePath'])
            cv2.imwrite(os.path.join(split[1], "Images", f"{data['patientID']}.png"), ds.pixel_array)
            data['FilePath'] = os.path.join("Images", f"{data['patientID']}.png")
            datasetJson.append(data)
        with open(os.path.join(split[1], "datasetInformation.json"), 'w') as jsonFile:
            json.dump(datasetJson, jsonFile)
