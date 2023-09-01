import argparse
import os
import time
from contextlib import nullcontext
from datetime import datetime
from statistics import fmean
from typing import Union

import numpy as np
import tensorboard
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet101
from tqdm import tqdm

from data.dataset import PneumoniaDetectionDataset
from model import PneumoniaDetectionModel


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--common", help="Path to common parent directory", required=True)
    p.add_argument("--device", default="cuda", help="Device to use for training")
    p.add_argument("--img_size", type=int, default=256, help="Image size to train the model at")
    p.add_argument("--epochs", type=int, default=40, help="Number of epochs to train for")
    p.add_argument("--val_freq", type=int, default=5, help="How many epochs between validations")
    p.add_argument("--save_frequency", type=int, default=5, help="How many epochs between model save")
    p.add_argument("--save_dir", default="checkpoints", help="Location of where to save model")
    p.add_argument("--load_model", type=str, default=None,
                   help="Location of where the model you want to load is stored")
    p.add_argument("--batch_size", type=int, help="Batch-size to train with", default=4)
    p.add_argument("--num_workers", type=int, help="Number of workers to use for dataloading", default=4)
    p.add_argument("--lr_step_size", type=int, default=30, help="How often to decay learning rate")
    p.add_argument("--lr_gamma", type=float, default=0.1, help="Factor to decrease learning rate by")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate to use in training")
    p.add_argument("--momentum", type=float, default=0.9, help="Momentum of learning rate for ADAM")
    p.add_argument("--name", type=str, default=f"Train_{datetime.now().strftime('%m_%d_%H_%M')}",
                   help="Name to save results under")

    args = p.parse_args()
    return args


def evalModel(model: resnet101, dataLoader: DataLoader, optimizer: torch.optim, args) -> Union[float, Figure]:
    """
    :param model: Model object to train
    :param dataLoader: Dataloader for desired dataset
    :param optimizer: Optimizer in use
    :param args: Copy of commandline arguments dictionary
    :return: Accuracy of evaluation
    """
    torch.cuda.empty_cache()
    time.sleep(2)
    print("Evaluating Model")
    time.sleep(2)
    correct = 0
    predictionList, labelList = [], []
    for data in tqdm(dataLoader):
        batch, label, lungClass = data['image'], data['label'], data['class']
        batch = batch.to(args.device)
        label = label.to(args.device)
        lungClass = lungClass.to(args.device)
        optimizer.zero_grad()

        # Use AMP/fp16
        with torch.amp.autocast(args.device) if args.device != "cpu" else nullcontext():
            with torch.no_grad():
                loss = model(batch)
        res = loss.cpu().argmax(-1)
        predictionList.extend(res.numpy())
        labelList.extend(np.argmax(label.cpu(), -1))
        correct += np.count_nonzero(res == np.argmax(label.cpu(), -1))

    confusionMatrix = confusion_matrix(labelList, predictionList)
    confusionMatrixImg = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=[True, False])
    confusionMatrixImg.plot()

    return correct / len(dataLoader.dataset), confusionMatrixImg.figure_


def trainEpoch(model: resnet101, dataLoader: DataLoader, optimizer: torch.optim,
               scaler: torch.cuda.amp.GradScaler, args) -> list:
    """
    :param model: Model object to train
    :param dataLoader: Dataloader for desired dataset
    :param optimizer: Optimizer in use
    :param scaler: Grad scaler in use
    :param args: Copy of commandline arguments dictionary
    :return: Running loss of training
    """
    runningloss = []
    for data in tqdm(dataLoader):
        batch, label, lungClass = data['image'], data['label'], data['class']
        batch = batch.to(args.device)
        label = label.to(args.device)
        lungClass = lungClass.to(args.device)
        optimizer.zero_grad()

        # Use AMP/fp16

        with torch.amp.autocast(args.device) if args.device != "cpu" else nullcontext():
            loss = model(batch)

        loss = loss.to(torch.float32)
        loss = criterion(loss, label)

        if args.device != "cpu":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
        runningloss.append(loss.item())
    return runningloss


if __name__ == '__main__':

    args = parseArgs()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    testSet = PneumoniaDetectionDataset(os.path.join(args.data, "Test", "datasetInformation.json"),
                                        imgSize=args.img_size)
    trainSet = PneumoniaDetectionDataset(os.path.join(args.data, "Train", "datasetInformation.json"),
                                         imgSize=args.img_size)
    valSet = PneumoniaDetectionDataset(os.path.join(args.data, "Test", "datasetInformation.json"),
                                       imgSize=args.img_size)
    # Setup datasets

    print(f"Found the following number of images:\nTrain: {len(trainSet)}\nVal: {len(valSet)}\nTest: {len(testSet)}")

    # Setup dataloaders

    trainSetDataLoader = DataLoader(trainSet, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)
    valSetDataLoader = DataLoader(valSet, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)
    testSetDataLoader = DataLoader(testSet, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    # Load model
    model = PneumoniaDetectionModel()
    if args.load_model and os.path.exists(args.load_model):
        model.load_state_dict(torch.load(args.load_model)["model"])
        print(f"Loaded {args.load_model}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = StepLR(optimizer, args.lr_step_size, gamma=args.lr_gamma)

    time.sleep(2)
    iterCount = 0
    print("Starting Tensorboard server")
    tensorBoardProgram = tensorboard.program.TensorBoard()
    tensorBoardProgram.configure(argv=[None, '--logdir', "tensorboard_runs"])
    url = tensorBoardProgram.launch()
    print(f"Launched Tensorboard server at {url}")
    tensorboardWriter = SummaryWriter(log_dir=f"tensorboard_runs/{args.name}")
    # tensorboardWriter.add_graph(model, torch.zeros((1,3,args.img_size,args.img_size)), use_strict_trace=True) # Broken with torch 1.12.1
    model.to(args.device)
    model.train()
    print(f"Starting train sequence for {args.epochs} epochs!")
    for epoch in range(1, args.epochs + 1, 1):
        print(f"Starting epoch {epoch}")
        torch.cuda.empty_cache()
        time.sleep(2)
        runningLoss = trainEpoch(model, trainSetDataLoader, optimizer, scaler, args)

        # Update learning rate
        scheduler.step()
        torch.cuda.empty_cache()
        if epoch % args.val_freq == 0:
            accuracy, plot = evalModel(model, valSetDataLoader, optimizer, args)
            print(f"Accuracy after validation is {accuracy * 100}%")
            # accuracy, plot = evalModel(model, testSetDataLoader, optimizer,args)  # Grouping test set here since validation set is much smaller, test set not used for any hyper-param optimizations
            # print(f"Accuracy after test is {accuracy * 100}%")
            tensorboardWriter.add_scalar("Val/Accuracy", accuracy, global_step=epoch)
            tensorboardWriter.add_figure("Val/Figure", plot, global_step=epoch)
            plot.savefig('confusion_matrix.png')

        if epoch % args.save_frequency == 0:
            saveDir = os.path.join(args.save_dir, f"{args.name}_model.pth")
            modelDict = {
                "model": model.state_dict(),
                "imgSize": args.img_size
            }
            torch.save(modelDict, saveDir)
            print(f"Saved Model to {saveDir}")
        tensorboardWriter.add_scalar("Train/Loss", fmean(runningLoss), global_step=epoch)

    tensorboardWriter.close()
    print('Finished Training')
