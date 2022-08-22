import argparse
import os
import time
from datetime import datetime
from statistics import fmean

import numpy as np
import tensorboard
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vision_transformer
from tqdm import tqdm

from data.dataset import PneumoniaDetectionDataset


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="Path to data parent directory", required=True)
    p.add_argument("--device", default="cuda", help="Device to use for training")
    p.add_argument("--img_size", type=int, default=64, help="Image size to train the model at")
    p.add_argument("--epochs", type=int, default=60, help="Number of epochs to train for")
    p.add_argument("--val_freq", type=int, default=10, help="How many epochs between validations")
    p.add_argument("--save_frequency", type=int, default=10, help="How many epochs between model save")
    p.add_argument("--save_dir", default="checkpoints", help="Location of where to save model")
    p.add_argument("--load_model", type=str, default=None,
                   help="Location of where the model you want to load is stored")
    p.add_argument("--batch_size", type=int, help="Batch-size to train with", default=4)
    p.add_argument("--num_workers", type=int, help="Number of workers to use for dataloading", default=4)
    p.add_argument("--lr_step_size", type=int, default=20, help="How often to decay learning rate")
    p.add_argument("--lr_gamma", type=float, default=0.1, help="Factor to decrease learning rate by")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate to use in training")
    p.add_argument("--momentum", type=float, default=0.9, help="Momentum of learning rate for ADAM")
    p.add_argument("--name", type=str, default=f"Train_{datetime.now().strftime('%m_%d_%H_%M_%S')}",
                   help="Name to save results under")

    args = p.parse_args()
    return args


def evalModel(model: vision_transformer, dataLoader: DataLoader, optimizer: torch.optim, args) -> float:
    """
    :param model: Model object to train
    :param dataLoader: Dataloader for desired dataset
    :param optimizer: Optimizer in use
    :param args: Copy of commandline arugments dictionary
    :return: Accuracy of evaluation
    """
    torch.cuda.empty_cache()
    time.sleep(1)
    print("Evaluating Model")
    correct = 0
    for data in tqdm(dataLoader):
        batch, label = data['image'], data['class']
        batch = batch.to(args.device)
        label = label.to(args.device)
        optimizer.zero_grad()

        # Use AMP/fp16
        with torch.amp.autocast(args.device):
            with torch.no_grad():
                loss = model(batch)

        res = loss.cpu().argmax(-1)
        correct += np.count_nonzero(res == np.argmax(label.cpu(), -1))

    return correct / len(dataLoader.dataset)


def trainEpoch(model: vision_transformer, dataLoader: DataLoader, optimizer: torch.optim,
               scaler: torch.cuda.amp.GradScaler, args) -> float:
    """
    :param model: Model object to train
    :param dataLoader: Dataloader for desired dataset
    :param optimizer: Optimizer in use
    :param scaler: Grad scaler in use
    :param args: Copy of commandline arugments dictionary
    :return: Running loss of training
    """
    runningloss = []
    for data in tqdm(dataLoader):
        batch, label = data['image'], data['class']
        batch = batch.to(args.device)
        label = label.to(args.device)
        optimizer.zero_grad()

        # Use AMP/fp16
        with torch.amp.autocast(args.device):
            loss = model(batch)

        loss = loss.to(torch.float32)
        loss = criterion(loss, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        runningloss.append(loss.item())
    return runningloss


if __name__ == '__main__':

    args = parseArgs()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    testSet = PneumoniaDetectionDataset(os.path.join(args.data, "test"), imgSize=args.img_size)
    trainSet = PneumoniaDetectionDataset(os.path.join(args.data, "train"), imgSize=args.img_size)
    valSet = PneumoniaDetectionDataset(os.path.join(args.data, "val"), imgSize=args.img_size)
    # Setup datasets

    print(f"Found the following number of images:\nTrain: {len(trainSet)}\nVal: {len(valSet)}\nTest: {len(testSet)}")

    # Setup dataloaders

    trainSetDataLoader = DataLoader(trainSet, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers)
    valSetDataLoader = DataLoader(valSet, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
    testSetDataLoader = DataLoader(testSet, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)

    # Load model
    model = vision_transformer.vit_b_16(num_classes=2, image_size=args.img_size)
    if args.load_model and os.path.exists(args.load_model):
        model.load_state_dict(torch.load(args.load_model)["model"])
        print(f"Loaded {args.load_model}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = StepLR(optimizer, args.lr_step_size, gamma=args.lr_gamma)

    time.sleep(1)
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
        time.sleep(1)
        runningLoss = trainEpoch(model, trainSetDataLoader, optimizer, scaler, args)

        # Update learning rate
        scheduler.step()

        torch.cuda.empty_cache()

        if epoch % args.val_freq == 0:
            accuracy = evalModel(model, valSetDataLoader, args)
            print(f"Accuracy after validation is {accuracy * 100}%")
            accuracy = evalModel(model, testSetDataLoader,
                                 args)  # Grouping test set here since validation set is much smaller, test set not used for any hyper-param optimizations

        print(f"Accuracy after test is {accuracy * 100}%")
        tensorboardWriter.add_scalar("Accuracy/Val", accuracy, global_step=epoch)

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
