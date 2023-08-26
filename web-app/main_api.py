import fastapi

import argparse
import uvicorn
import torch
from typing import Union
from model import PneumoniaDetectionModel
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from data.dataset import PneumoniaDetectionDataset
import logging
import io
from PIL import Image


app = FastAPI()


@app.get("/")
def read_root():
    return {"Title": "Pneumonia Detection API"}




@app.post("/items")
async def read_item(file: UploadFile ):
    contents = await file.read()
    logging.info(contents)

    image = Image.open(io.BytesIO(contents))
    image.save('test.png')
    img = PneumoniaDetectionDataset.basicPreprocess(modelData['imgSize'])(image)
    img = img.type(torch.FloatTensor).unsqueeze(0).to(device)
    result = model(img)
    outputClass = PneumoniaDetectionDataset.getClassMap()[result.argmax().item()]
    printMessage = f"The patient has {outputClass}" if outputClass == "Pneumonia" else f"The patient is healthy!"

    return {"item_id": 'item_id', "q": printMessage}


# def parseArgs():
#     p = argparse.ArgumentParser()
#     p.add_argument("--device", default="cpu", help="Device to use for training")
#     p.add_argument("--load_model", type=str, default="static/prod.pth",
#                    help="Location of where the model you want to load is stored")
#     arguments = p.parse_args()
#     return arguments


def loadModel():
    # args = parseArgs()
    # load_model = args.load_model
    # device = args.device

    load_model = "static/prod.pth"
    device = "cpu"

    modelData = torch.load(load_model, map_location=torch.device('cpu'))
    model = PneumoniaDetectionModel()
    model.load_state_dict(modelData['model'])
    print(f"Loaded {load_model}")
    model.to(device)
    model.eval()
    return model, modelData, device


model, modelData, device = loadModel()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)