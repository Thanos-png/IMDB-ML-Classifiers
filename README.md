# Text Classification Using AdaBoost, Random Forest, and RNNs (IMDB Dataset)

This project implements an **AdaBoost** and a **Random Forest** model as well as a **Stacked Bidirectional RNN with GRU cells** classifiers to classify movie reviews into **positive** or **negative** opinions. The dataset used is the **Large Movie Review Dataset (IMDB Dataset)**.

## Project Overview

### **Objective**
- Represent text as **binary feature vectors**, where each feature corresponds to the presence (`1`) or absence (`0`) of a word in the review.
- Construct a vocabulary by **removing the `n` most frequent and `k` rarest words** and selecting the **top `m` words with the highest information gain**.
- Evaluate the classifier on a **subset of training data (development set)** and **test data**.
- **GPU acceleration** is used to speed up training and prediction.

#### This project uses the **IMDB Large Movie Review Dataset**, available at:
- [Stanford AI IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [PyTorch IMDB Dataset](https://pytorch.org/text/stable/datasets.html#imdb)

## Installation
Clone this repository:
```bash
git clone https://github.com/Thanos-png/IMDB-ML-Classifiers.git
cd IMDB-ML-Classifiers
pip install -r requirements.txt
```

### Ensure GPU is Available
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `True` your GPU is ready.If `False` check your CUDA installation or the CPU will be used automatically.

## Running the Project
### Train the AdaBoost Model
```bash
cd src/
python train_adaboost.py
```
#### This will:
- Load and preprocess the IMDB dataset.
- Construct a **vocabulary** by removing frequent/rare words and selecting words based on **information gain**.
- Convert reviews into **binary feature vectors**.
- Train an **AdaBoost classifier** with **T=200** boosting iterations.

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

Custom Model and vocabulary saved to ../results/adaboost_model.pkl and ../results/vocab.pkl
Sklearn AdaBoost model saved to ../results/sklearn_adaboost.pkl

Running learning curve experiment (evaluating for positive class)...
(Table)

--- Best Hyperparameters ---
{'T': 200, 'm': 5000, 'n_most': 50, 'k_rarest': 50}
```

### Test the Model
```bash
cd src/
python test_adaboost.py
```
#### This will:
- Load the **trained adaboost model** and **vocabulary**.
- Convert **test reviews** into **binary feature vectors**.
- Make predictions and compute **test accuracy**.
- Evaluate both **custom AdaBoost and Sklearn’s AdaBoost** models.
- Compute and display **precision, recall, and F1-score** for **positive and negative** sentiment classes.
- Report **micro- and macro-averaged** evaluation metrics for both models. 

#### Expected output:
```
Loading test data...
Loaded 25000 test examples.
Test Accuracy: 82.37%

Loaded trained Sklearn AdaBoost model from ../results/sklearn_adaboost.pkl
Sklearn AdaBoost Test Accuracy: 80.77%

Custom AdaBoost Test Evaluation Metrics:
Category   Precision  Recall     F1        
Positive   0.8098     0.8462     0.8276    
Negative   0.8390     0.8012     0.8197    

Sklearn AdaBoost Test Evaluation Metrics:
Category   Precision  Recall     F1        
Positive   0.7875     0.8428     0.8142    
Negative   0.8309     0.7726     0.8007    

Custom AdaBoost Micro-averaged: Precision: 0.8237, Recall: 0.8237, F1: 0.8237
Custom AdaBoost Macro-averaged: Precision: 0.8244, Recall: 0.8237, F1: 0.8236

Sklearn AdaBoost Micro-averaged: Precision: 0.8077, Recall: 0.8077, F1: 0.8077
Sklearn AdaBoost Macro-averaged: Precision: 0.8092, Recall: 0.8077, F1: 0.8075
```

### Train the Stacked Bidirectional RNN Model
```bash
cd src/
python train_rnnmodel.py
```
#### This will:
- Load and preprocess the IMDB dataset.
- Construct a **vocabulary** and convert reviews into **sequences of token indices**.
- Use **pre-trained word embeddings** to initialize the embedding layer.
- Train a **Stacked Bidirectional RNN with GRU cells** using **Adam optimizer**.
- Monitor **training and development loss** across epochs.

#### Hyperparameters used:
| Parameter       | Value  | Description                                  |
| --------------- | ------ | -------------------------------------------- |
| `embedding_dim` | `300`  | Dimension of the pre-trained word embeddings |
| `hidden_dim`    | `256`  | Hidden dimension of the RNN                  |
| `num_layers`    | `2`    | Number of stacked RNN layers                 |
| `dropout`       | `0.2`  | Dropout probability                          |
| `num_epochs`    | `10`   | Number of epochs                             |
| `lr`            |`0.0002`| Learning rate                                |

#### Expected output:
```
Loading training data...
Loaded 25000 training examples.

--- Training with embedding_dim=300, hidden_dim=256, num_layers=2, dropout=0.2, lr=0.0002, num_epochs=10 ---
(Iterations)
Dev Accuracy: 87.00%

RNN Model and vocabulary saved to ../results/rnn_model.pth and ../results/vocab.pkl

--- Best Hyperparameters ---
{'embedding_dim': 300, 'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.0002, 'num_epochs': 10}
```

### Test the Model
```bash
cd src/
python test_rnnmodel.py
```
#### This will:
- Load the **saved RNN model** and **vocabulary**.
- Preprocess the **IMDB test dataset**, converting reviews into **sequences of token indices**.
- Evaluate the model on **test data** using **mini-batches** to prevent memory issues.
- Compute and display **test accuracy**.
- Calculate **precision, recall, and F1-score** for both **positive and negative** sentiment classes.
- Report **micro- and macro-averaged** evaluation metrics.

#### Expected output:
```
Loading test data...
Loaded 25000 test examples.
Test Accuracy: 86.91%

Evaluation Metrics on Test Data:
Category   Precision  Recall     F1        
Positive   0.8686     0.8698     0.8692    
Negative   0.8697     0.8684     0.8690    

Micro-averaged: Precision: 0.8691, Recall: 0.8691, F1: 0.8691
Macro-averaged: Precision: 0.8691, Recall: 0.8691, F1: 0.8691
```

## Key Implementations
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

### Stacked Bidirectional RNN Implementation
- Uses a **Stacked Bidirectional GRU** network for sequential text classification.
- Each review is represented as a **sequence of token indices**, processed using **pre-trained word embeddings**.
- The **GRU layers** capture contextual dependencies in both forward and backward directions.
- A **global max pooling layer** extracts the most important features from the hidden states.
- The final classification is performed using a **fully connected layer with softmax activation**.

## Results & Analysis
### Custom adaboost metrics
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

## Credits & Acknowledgements
- IMDB Dataset: [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/)

##### Note: This project has been tested with Python 3.12 but should work with other 3.x versions.
