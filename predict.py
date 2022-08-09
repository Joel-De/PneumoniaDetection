import argparse
import os

import torch
from PIL import Image
from torchvision.models import vision_transformer

from data.dataset import PneumoniaDetectionDataset


def parseArgs():
    p = argparse.ArgumentParser()

    p.add_argument("--data", help="Path to image", required=True)
    p.add_argument("--device", default="cuda", help="Device to use for training")
    p.add_argument("--save_dir", default="runs", help="Location of where to save model")
    p.add_argument("--load_model", type=str, default=None, required=True,
                   help="Location of where the model you want to load is stored")
    args = p.parse_args()
    return args


if __name__ == '__main__':
    args = parseArgs()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    modelData = torch.load(args.load_model)
    model = vision_transformer.vit_b_16(num_classes=2, image_size=modelData['imgSize'])

    model.load_state_dict(modelData['model'])
    print(f"Loaded {args.load_model}")
    model.to(args.device)
    model.eval()

    img = Image.open(args.data).convert('RGB')
    img = PneumoniaDetectionDataset.basicPreprocess(modelData['imgSize'])(img)
    img = img.type(torch.FloatTensor).unsqueeze(0).to(args.device)
    result = model(img)
    outputClass = PneumoniaDetectionDataset.getClassMap()[result.argmax().item()]
    print(f"The class of the image is {outputClass}")
