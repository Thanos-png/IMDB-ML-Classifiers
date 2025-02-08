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
python src/train.py
```
#### This will:
- Load and preprocess the IMDB dataset.
- Construct a **vocabulary** by removing frequent/rare words and selecting words based on **information gain**.
- Convert reviews into **binary feature vectors**.
- Train an **AdaBoost classifier** with **T=150** boosting iterations.

#### Hyperparameters used:
| Parameter | Value | Description                   |
| --------- | ----- | ----------------------------- |
| `T`       | `150` | Number of AdaBoost iterations |
| `m`       | `3000`| Vocabulary size               |
| `n_most`  | `50`  | Most frequent words removed   |
| `k_rarest`| `50`  | Rarest words removed          |

#### Expected output:
```
Development Accuracy: 80.96%
```

### Test the Model
```
python src/test.py
```
#### This will:
- Load the **trained model** and **vocabulary**.
- Convert **test reviews** into **binary feature vectors**.
- Make predictions and compute **test accuracy**.

#### Expected output:
```
Test Accuracy: 81.61%
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

## 🏆 Credits & Acknowledgements
- IMDB Dataset: [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/)
