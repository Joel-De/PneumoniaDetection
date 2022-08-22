import argparse
import io

import torch
from PIL import Image
from flask import Flask, render_template
from flask import request, redirect
from torchvision.models import vision_transformer

from data.dataset import PneumoniaDetectionDataset

app = Flask(__name__)
app.secret_key = b'@VOv3oactreto8yavheE$B^eo'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img.save("static/tmp.png")
        img = PneumoniaDetectionDataset.basicPreprocess(modelData['imgSize'])(img)
        img = img.type(torch.FloatTensor).unsqueeze(0).to(args.device)
        result = model(img)
        outputClass = PneumoniaDetectionDataset.getClassMap()[result.argmax().item()]
        printMessage = f"The patient has {outputClass}" if outputClass == "Pneumonia" else f"The patient is healthy!"
        return render_template('display.html', user_image="static/tmp.png", result=printMessage)
    return render_template('index.html')


def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", help="Device to use for training")
    p.add_argument("--load_model", type=str, default=None, required=True,
                   help="Location of where the model you want to load is stored")
    arguments = p.parse_args()
    return arguments


if __name__ == '__main__':
    args = parseArgs()
    modelData = torch.load(args.load_model)
    model = vision_transformer.vit_b_16(num_classes=2, image_size=modelData['imgSize'])
    model.load_state_dict(modelData['model'])
    print(f"Loaded {args.load_model}")
    model.to(args.device)
    model.eval()
    app.run(debug=True)
