# AdaBoost for Text Classification (IMDB Dataset)

This project implements the **AdaBoost** algorithm with **decision stumps** to classify movie reviews into **positive** or **negative** opinions. The dataset used is the **Large Movie Review Dataset (IMDB Dataset)**.

## 🚀 Project Overview

### **Objective**
- Implement **AdaBoost** using **decision stumps** as weak classifiers.
- Represent text as **binary feature vectors**, where each feature corresponds to the presence (`1`) or absence (`0`) of a word in the review.
- Construct a vocabulary by **removing the `n` most frequent and `k` rarest words** and selecting the **top `m` words with the highest information gain**.
- Evaluate the classifier on a **subset of training data (development set)** and **test data**.
- **GPU acceleration** is used to speed up training and prediction.

#### This project uses the **IMDB Large Movie Review Dataset**, available at:
- [Stanford AI IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [PyTorch IMDB Dataset](https://pytorch.org/text/stable/datasets.html#imdb)

Before running the code, **ensure the dataset is downloaded and placed in `data/aclImdb/`**.

## Installation
Clone this repository:
```
git clone https://github.com/Thanos-png/AdaboostTextClassifier.git
cd AdaboostTextClassifier
pip install -r requirements.txt
```

### Ensure GPU is Available
```
python -c "import torch; print(torch.cuda.is_available())"
```
If `True` your GPU is ready.
If `False` check your CUDA installation or use CPU.

## Running the Project
### Train the AdaBoost Model
```
cd src/
python train.py
```
#### This will:
- Load and preprocess the IMDB dataset.
- Construct a **vocabulary** by removing frequent/rare words and selecting words based on **information gain**.
- Convert reviews into **binary feature vectors**.
- Train an **AdaBoost classifier** with **T=150** boosting iterations.

#### Hyperparameters used:
| Parameter | Value | Description                   |
| --------- | ----- | ----------------------------- |
| `T`       | `200` | Number of AdaBoost iterations |
| `m`       | `5000`| Vocabulary size               |
| `n_most`  | `50`  | Most frequent words removed   |
| `k_rarest`| `50`  | Rarest words removed          |

#### Expected output:
```
Loading training data...
Loaded 25000 training examples.

--- Training with T=200, m=5000, n_most=50, k_rarest=50 ---
Building vocabulary...
Vectorizing texts...
Training AdaBoost classifier...
(Iterations)
Development Accuracy: 82.60%
Training Sklearn AdaBoost classifier...
Sklearn AdaBoost Dev Accuracy: 80.90%
Model and vocabulary saved to ../results/adaboost_model.pkl and ../results/vocab.pkl

Running learning curve experiment (evaluating for positive class)...
(Table)

--- Best Hyperparameters ---
{'T': 200, 'm': 5000, 'n_most': 50, 'k_rarest': 50}
```

### Test the Model
```
cd src/
python test.py
```
#### This will:
- Load the **trained model** and **vocabulary**.
- Convert **test reviews** into **binary feature vectors**.
- Make predictions and compute **test accuracy**.

#### Expected output:
```
Loading test data...
Loaded 25000 test examples.
Test Accuracy: 82.37%

Evaluation Metrics on Test Data:
Category   Precision  Recall     F1        
Positive   0.8098     0.8462     0.8276    
Negative   0.8390     0.8012     0.8197    

Micro-averaged: Precision: 0.8237, Recall: 0.8237, F1: 0.8237
Macro-averaged: Precision: 0.8244, Recall: 0.8237, F1: 0.8236
```

## 🛠️ Key Implementations
### Text Preprocessing (Binary Feature Representation)
- **Tokenization**: Splitting text into words.
- **Vocabulary Selection**:
  - Remove `n` **most frequent** and `k` **rarest words**.
  - Select `m` **words with highest information gain**.
- **Vectorization**:
  - Convert reviews into **binary feature vectors** (`1` = word present, `0` = word absent).

### AdaBoost Implementation
- Uses **decision stumps** as weak learners.
- Each weak learner classifies reviews based on the presence/absence of a single word.
- Weighted voting of multiple weak learners improves accuracy.

## 📈 Results & Analysis
| Size  | Train Prec | Train Rec | Train F1 | Dev Prec | Dev Rec | Dev F1 |
| ----- | ---------- | --------- | -------- | -------- | ------- | ------ |
| 2000  | 0.8365     | 0.8797    | 0.8576   | 0.7699   | 0.8423  | 0.8045 |
| 4000  | 0.8239     | 0.8600    | 0.8416   | 0.7815   | 0.8483  | 0.8135 |
| 8000  | 0.8236     | 0.8395    | 0.8315   | 0.8048   | 0.8302  | 0.8173 |
| 12000 | 0.8273     | 0.8429    | 0.8350   | 0.8048   | 0.8318  | 0.8181 |
| 16000 | 0.8251     | 0.8437    | 0.8343   | 0.8050   | 0.8294  | 0.8170 |
| 20000 | 0.8176     | 0.8528    | 0.8349   | 0.7992   | 0.8491  | 0.8234 |

### Custom vs Sklearn adaBoost comparison
| Size  | Custom Train F1 | Custom Dev F1 | Sklearn Train F1 | Sklearn Dev F1 |
| ----- | --------------- | ------------- | ---------------- | -------------- |
| 2000  | 0.8564          | 0.8067        | 0.8483           | 0.7970         |
| 4000  | 0.8515          | 0.8137        | 0.8331           | 0.8068         |
| 8000  | 0.8378          | 0.8160        | 0.8272           | 0.8092         |
| 12000 | 0.8334          | 0.8191        | 0.8205           | 0.8081         |
| 16000 | 0.8320          | 0.8189        | 0.8196           | 0.8115         |
| 20000 | 0.8327          | 0.8272        | 0.8190           | 0.8132         |

#### The visualizations are stored in the `results/plots` directory.

## 🏆 Credits & Acknowledgements
- IMDB Dataset: [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/)
