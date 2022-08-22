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
        self.maxDatasetLen = None
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

        imgClass = [0, 0]
        imgClass[self.datasetInformation[idx]['Label']] = 1
        imgClass = torch.tensor(imgClass, dtype=torch.float)
        packedData = {'image': img, 'class': imgClass}
        return packedData


if __name__ == '__main__':
    dat = PneumoniaDetectionDataset(r"F:\PycharmProjects\PneumoniaDetection\data\RSNADataset\Train\datasetInformation.json")
    print(dat.__len__())
    print(dat.__getitem__(200)['image'].size())
