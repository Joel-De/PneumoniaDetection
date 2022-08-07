import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vision_transformer
from tqdm import tqdm

from data.dataset import PneumoniaDetectionDataset

if __name__ == '__main__':

    # REQUIRED PARAMS

    datasetParentDir = r""
    batchSize = 2
    device = "cuda:0"
    imgSize = 16 * 18

    # todo
    epochs = 50
    valFreq = 20

    testSet = PneumoniaDetectionDataset(os.path.join(datasetParentDir, "test"), imgSize=imgSize)
    trainSet = PneumoniaDetectionDataset(os.path.join(datasetParentDir, "train"), imgSize=imgSize)
    valSet = PneumoniaDetectionDataset(os.path.join(datasetParentDir, "val"), imgSize=imgSize)
    # Setup datasets

    print(f"Found the following number of images:\nTrain: {len(trainSet)}\nVal: {len(valSet)}\nTest: {len(testSet)}")

    # Setup dataloaders

    trainSetDataLoader = DataLoader(trainSet, batch_size=batchSize,
                                    shuffle=True, num_workers=2)
    valSetDataLoader = DataLoader(valSet, batch_size=batchSize,
                                  shuffle=True, num_workers=2)
    testSetDataLoader = DataLoader(testSet, batch_size=batchSize,
                                   shuffle=False, num_workers=2)

    # Load model
    model = vision_transformer.vit_b_16(num_classes=2, image_size=imgSize)
    model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()

    time.sleep(1)
    iterCount = 0


    def evalModel(dataLoader: DataLoader):
        """
        :param dataLoader: DataLoader instance of the set you wish to evaluate on
        :type dataLoader: Dataloader
        :return: Accuracy of model on the data
        :rtype: float
        """
        time.sleep(1)
        print("Evaluating Model")
        correct = 0
        model.eval()
        for data in tqdm(dataLoader):
            batch, label = data['image'], data['class']
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            # Use AMP/fp16
            with torch.cuda.amp.autocast():
                loss = model(batch)

            res = loss.cpu().argmax(-1)
            correct += np.count_nonzero(res == np.argmax(label.cpu(), -1))

        return correct / len(dataLoader.dataset)


    print(f"Starting train sequence for {epochs} epochs!")
    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        running_loss = 0.0
        time.sleep(1)
        for data in tqdm(trainSetDataLoader):
            batch, label = data['image'], data['class']
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            # Use AMP/fp16
            with torch.cuda.amp.autocast():
                loss = model(batch)

            loss = loss.to(torch.float32)
            loss_ = criterion(loss, label)

            scaler.scale(loss_).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.save(model.state_dict(), "Model.pth")
        accuracy = evalModel(valSetDataLoader)
        print(f"Accuracy after validation is {accuracy}%")
        accuracy = evalModel(testSetDataLoader)
        print(f"Accuracy after test is {accuracy}%")
    print('Finished Training')
