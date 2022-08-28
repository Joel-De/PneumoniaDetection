import os
import torch.nn
import torchvision
import json

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union

class PneumoniaDetectionDataset(Dataset):
    def __init__(self, datasetInformationDir: Union[str, Path], imgSize=224, transform=None):
        """
        :param datasetInformationDir: Directory of root dataset
        :param transform:#NOT IMPLEMENTED
        """
        self.rootDir = os.path.join(datasetInformationDir, "..")
        self.maxDatasetLen = 100# None
        with open(datasetInformationDir, 'r') as jsonFile:
            self.datasetInformation = json.load(jsonFile)
        self.transform = transform
        self.imgSize = imgSize

    @staticmethod
    def getClassMap() -> dict:
        """
        :return: Returns dictionary of index -> ID
        """
        return {0: 'Normal', 1: "Pneumonia"}

    @staticmethod
    def getLungClassMap() -> dict:
        """
        :return: Returns dictionary of index -> ID
        """
        return {"No Lung Opacity / Not Normal":0,
                "Normal":1,
                "Lung Opacity":2}

    @staticmethod
    def basicPreprocess(imgSize: int) -> torchvision.transforms:
        """
        :param imgSize: Size of desired output image
        :return: Torchvision transform module
        """
        return torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((imgSize, imgSize))])

    def __len__(self) -> int:
        return min(self.maxDatasetLen, len(self.datasetInformation)) if self.maxDatasetLen else len(self.datasetInformation)

    def __getitem__(self, idx: int) -> dict[str:torch.Tensor]:
        """
        :param idx: Index of data to use
        :return: Dictionary of image and class label
        """
        img = Image.open(os.path.join(self.rootDir, self.datasetInformation[idx]['FilePath'])).convert('RGB')
        if self.transform:
            img = self.transform(img)

        img = self.basicPreprocess(self.imgSize)(img)
        img = img.type(torch.FloatTensor)

        imgLabel = [0, 0]
        imgLabel[self.datasetInformation[idx]['Label']] = 1
        imgLabel = torch.tensor(imgLabel, dtype=torch.float)

        lungClass = [0,0,0]
        lungClass[self.getLungClassMap()[self.datasetInformation[idx]['AdditionalInformation']]] = 1
        lungClass = torch.tensor(lungClass, dtype=torch.float)
        packedData = {'image': img, 'label': imgLabel, 'class':lungClass}
        return packedData


if __name__ == '__main__':
    dat = PneumoniaDetectionDataset(r"F:\PycharmProjects\PneumoniaDetection\RSNADataset\Train\datasetInformation.json")
    print(dat.__len__())
    print(dat.__getitem__(200))
