# Credit Card Fraud Detection using Machine Learning

A beginner-friendly end-to-end ML project that classifies card transactions as **legitimate** or **fraudulent**.

## Features
- Loads a transaction CSV dataset (or creates a small synthetic demo dataset)
- Exploratory data analysis (class distribution and amount distribution)
- Preprocessing with a train/test split and feature scaling
- Trains Logistic Regression and Random Forest models
- Evaluates precision, recall, F1-score, ROC-AUC and confusion matrix
- Saves the trained model and evaluation charts
- Includes a Streamlit web interface for single-transaction prediction

> **Important:** The included synthetic dataset is only for demonstrating that the project runs. It is not representative of real banking fraud and its metrics must not be presented as real-world performance. For meaningful evaluation, use the public dataset described below.

## Requirements
Python 3.10 or newer is recommended.

## Run in Jupyter or terminal

1. Extract the ZIP.
2. Open a terminal in this project folder.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Run the training script:

```bash
python train.py
```

5. View charts and saved model in `outputs/`.

## Run the web app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Use a real dataset (recommended)

Download the **Credit Card Fraud Detection** dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` inside the `data/` folder. Then run:

```bash
python train.py --data data/creditcard.csv
```

The common Kaggle dataset has columns `Time`, `V1` ... `V28`, `Amount`, and `Class`, where `Class=1` indicates fraud. The code handles this schema.

## Project structure

```text
Credit_Card_Fraud_Detection/
├── app.py
├── train.py
├── fraud_detection.ipynb
├── requirements.txt
├── README.md
├── data/
│   └── README.txt
└── outputs/
    └── (generated after training)
```

## Academic integrity and interpretation
- Do not claim the synthetic demo results represent actual fraud detection performance.
- Fraud datasets are highly imbalanced. Accuracy alone is misleading; focus on fraud-class precision, recall, F1-score, PR-AUC/ROC-AUC and the confusion matrix.
- This is an educational prototype, not a production banking system. It does not connect to a bank, process payments, or make real financial decisions.
