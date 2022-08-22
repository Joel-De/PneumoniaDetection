import os
import torch.nn
import torchvision

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union

class PneumoniaDetectionDataset(Dataset):
    def __init__(self, rootDir: Union[str, Path], imgSize=224, transform=None):
        """
        :param rootDir: Directory of root dataset
        :param transform:#NOT IMPLEMENTED
        """
        self.rootDir = rootDir
        self.maxDatasetLen = None
        self.imgListNormal = [os.path.join(self.rootDir, "NORMAL", fName) for fName in
                              os.listdir(os.path.join(self.rootDir, "NORMAL"))]
        self.imgListPneumonia = [os.path.join(self.rootDir, "PNEUMONIA", fName) for fName in
                                 os.listdir(os.path.join(self.rootDir, "PNEUMONIA"))]
        self.combinedImages = [[img, 0] for img in self.imgListNormal] + [[img, 1] for img in self.imgListPneumonia]
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
        return min(self.maxDatasetLen, len(self.combinedImages)) if self.maxDatasetLen else len(self.combinedImages)

    def __getitem__(self, idx: int) -> dict[str:torch.Tensor]:
        """
        :param idx: Index of data to use
        :return: Dictionary of image and class label
        """
        img = Image.open(self.combinedImages[idx][0]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        img = self.basicPreprocess(self.imgSize)(img)
        img = img.type(torch.FloatTensor)

        imgClass = [0, 0]
        imgClass[self.combinedImages[idx][1]] = 1
        imgClass = torch.tensor(imgClass, dtype=torch.float)
        packedData = {'image': img, 'class': imgClass}
        return packedData


if __name__ == '__main__':
    dat = PneumoniaDetectionDataset(r"../chest_xray/test")
    print(dat.__len__())
    print(dat.__getitem__(200)['image'].size)
