# [BirdCLEF 2024](https://www.kaggle.com/competitions/birdclef-2024/overview) – 20th Place Solution

## 0. Environment

Training was conducted on [Vast.ai](https://vast.ai/) cloud GPU instances

Run the following command to install dependencies.

```sh
pip3 install -r requirements.txt
```

## 1. Data Preparation

### train metadata, audios

Download the official **BirdCLEF 2024** data from the competition page:  
https://www.kaggle.com/competitions/birdclef-2024/data

- Place `train_metadata.csv` into `./inputs/`
- Extract all training audio files into `./inputs/train_audios/`  

#### Optional: Additional recordings from Xeno-canto
You may enrich training data with extra recordings from [Xeno-canto](https://xeno-canto.org).  
When adding external audio, convert to **mono**, **32 kHz**, **OGG** for consistency. Some referenced files may be unavailable if uploaders removed them.

### Background noise (optional, for augmentation)

Download background noise clips from:  
https://www.kaggle.com/datasets/honglihang/background-noise

Place files under: `./inputs/background_noise/`

## 2. Training

Before running the training, export your WANDB_API_KEY environment variable instead of editing configs.

Training can be initialized with:

```sh
# if you want to use pseudo as target label, specify the option --use_pseudo
# pseudo label doesn't improve the score and is a vulnerable point in training pipeline
python3 train.py --stage STAGE --model_name MODEL_NAME
```

After training, the last checkpoint (model weights) will be saved to the folder ./outputs/MODEL_NAME/pytorch/STAGE

The model’s training and evaluation results can be checked on wandb, as shown in the figure below.
<img width="729" height="242" alt="image" src="https://github.com/user-attachments/assets/237b1d7a-07f9-4a6d-acb3-ea1270dd3b69" />


## 3. Model Conversion

Run

```sh
python3 convert.py --model_name MODEL_NAME
```

The onnx model will be saved to the folder ./outputs/MODEL_NAME/onnx.

The openvino model will be saved to the folder ./outputs/MODEL_NAME/openvino.

## 4. Inference

Inference is published in a kaggle kernel [here](https://www.kaggle.com/code/sugar0/birdclef2024-inference)
