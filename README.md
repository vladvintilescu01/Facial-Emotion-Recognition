# Facial-Emotion-Recognition
This repository contains the software architecture for Facial-Emotion-Recognition deep‑learning models ( DenseNet121, custom CNN)


---

## Prerequsits

Follow these steps if you want to train inside the Anaconda ecosystem. Also, if you want to see the demo of this app you will need to do these steps. You can use any type of environment to train these models, the steps will look almost the same.

### 1. Install Anaconda

Download and install Anaconda from the official page:  
<https://www.anaconda.com/products/distribution>

### 2. Create a New Environment

Open **Anaconda Prompt** (or your terminal) and create an isolated environment:

```bash
conda create -n DL-FER python=3.8
conda activate DL-FER
```

### 3. Install Required Libraries

With the environment active, install all necessary packages:

```bash
conda install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv
```

> *Alternative:* use pip  
> ```bash
> pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python
> ```

### 4. Install & Launch Spyder

If Spyder IDE is not already present:

```bash
conda install spyder
```

Then start it:

```bash
spyder
```

### 5. Get the dataset

The dataset is available via this link:https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge. After the download of this dataset you will need to create a folder, named 'dataset' and to put here the dataset. This dataset is created by me and it is created in collaboration with a teacher for a better result.

## How to train a model

1. In **Spyder**, select one of the models (e.g.:models/model_DenseNet121).  
2. Hit **Run** in Spyder to start training.  
3. Monitor the console for metrics, losses, and any early‑stopping callbacks.

---

## How to use the demo app

1. In **Spyder**, select the predict app (models/predict_real_time).  
1. Select one of already weights that exists(DenseNet or CustomCNN) or you can use one of your weights created by you. (e.g.:models/weights/DenseNet121_FER2013.h5)
2. Hit **Run** in Spyder to start the demo app.  
3. Right now the app can recognize: angry, happy, sad, surprise, neutral.