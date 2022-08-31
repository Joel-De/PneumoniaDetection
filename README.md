# Pneumonia Detection
Pneumonia is a common illness affecting millions of people around the world. Detecting whether a patient has pneumonia can be a time-consuming process from the time X-Ray images have been taken to when a qualified doctor can analyze them. This repository aims to provide an automated approach of this using deep learning. The input data are scans of X-Ray images of peoples chest, and  the model will determine whether the patient has Pneumonia or not. 

# Installation

To install this library create a new virtual environment with your package manager of choice and run the following command
```commandline
pip install -r requirements.txt
```
This will install all required packages required for this repository to work

# Dataset
The dataset being used in this repository is from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data?select=stage_2_train_images). This dataset contains ~30 000 images in total, since the test ground truth is not publicly available there is a utility script to split the original train dataset into a train split and test split.
```commandline
python process_dataset.py --data_dir "Path/To/Raw/Dataset/" --output_dir "Output/Dir/RNSADataset"
```

# Train

To train the model run the following command, additional arguments can be passed to further modify the training configuration, the following is a sample command that can be run to train the model. Typically, the default parameters will suffice for training the model. 
```commandline
python train.py --data "Path/To/Formatted/Dataset/" --output_dir "Output/Dir/RNSADataset" --epochs 100 --val_freq 10
```

# Inference

To run inference on a given image once the model has been trained run the following command, the script will print the result to console.
```commandline
python predict.py --data "Path/To/Image.png" --device "cuda" --load_model "Path/To/Model.pth"
```

# Hosting Web App

run web-app/application.py to host a flask server that provides a UI for uploading and running inference on images. Run the following command to launch the application
```commandline
python application.py --device "cpu" --load_model "Path/To/Model.pth"
```

# Tensorboard Bug
If on Windows and tensorboard is rendering as a white page, follow the instructions present [here](https://github.com/tensorflow/tensorboard/issues/3117#issuecomment-605531669), this is a bug with tensorflow, this should work correctly on UNIX systems.
