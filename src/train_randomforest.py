import os
import random
import numpy as np
import pickle
import torch
import datetime
import matplotlib.pyplot as plt
from preprocess import load_imdb_data, build_vocabulary, vectorize_texts
from randomforest import randomforest_train, randomforest_predict
from utils import to_tensor, compute_metrics_for_class_sklearn, compute_metrics_for_class_torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def plot_learning_curve_A(train_sizes, train_prec, train_rec, train_f1, dev_prec, dev_rec, dev_f1):
    """Plots precision, recall, and F1-score for train and dev sets as training size increases."""

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_f1, marker='o', linestyle='-', label="Train F1-score", color='blue')
    plt.plot(train_sizes, dev_f1, marker='s', linestyle='-', label="Dev F1-score", color='green')
    plt.plot(train_sizes, train_prec, marker='^', linestyle='-', label="Train Precision", color='red')
    plt.plot(train_sizes, dev_prec, marker='v', linestyle='-', label="Dev Precision", color='orange')
    plt.plot(train_sizes, train_rec, marker='d', linestyle='-', label="Train Recall", color='purple')
    plt.plot(train_sizes, dev_rec, marker='x', linestyle='-', label="Dev Recall", color='brown')

    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.title("Learning Curve: Train vs Dev Metrics")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plots_dir = os.path.join('..', 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'train_1_RF.png'))
    print(f"Plot saved at {os.path.join(plots_dir, 'train_1_RF.png')}")


def plot_learning_curve_B(train_sizes, train_f1, dev_f1, sklearn_train_f1, sklearn_dev_f1):
    """Plots F1-score learning curves for both Custom and Sklearn Random Forest."""

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_f1, marker='o', linestyle='-', label="Custom Train F1", color='blue')
    plt.plot(train_sizes, dev_f1, marker='s', linestyle='-', label="Custom Dev F1", color='green')
    plt.plot(train_sizes, sklearn_train_f1, marker='^', linestyle='--', label="Sklearn Train F1", color='red')
    plt.plot(train_sizes, sklearn_dev_f1, marker='v', linestyle='--', label="Sklearn Dev F1", color='orange')
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve: Custom vs Sklearn Random Forest(Positive Class)")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plots_dir = os.path.join('..', 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'train_2_RF.png'))
    print(f"Plot saved at {os.path.join(plots_dir, 'train_2_RF')}")


def main():
    # Start time
    start_time = datetime.datetime.now()
    print("Start time:", start_time,"\n")

    # Hyperparameters
    n_estimators_values = [700]  # Number of trees in the forest - Checked [10,50,150,200,300,400,500,,700,800,900]
    m_values = [5000]  # Vocabulary size - Checked [1000,2000,3000,4000,5000,7000,10000]
    n_most_values = [50]  # Most frequent words removed - Checked [20,50,100]
    k_rarest_values = [50]  # Rarest words removed - Checked [20,50,100]

    best_acc = 0
    best_params = {}

    # Load Training Data
    print("Loading training data...")
    root = os.path.join("..", "data", "aclImdb")
    texts, labels = load_imdb_data(split='train', root=root)    
    print(f"Loaded {len(texts)} training examples.")

    # Split into Training and Development Sets (80/20 split)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split_index = int(0.8 * len(texts))
    train_indices = indices[:split_index]
    dev_indices = indices[split_index:]

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    dev_texts = [texts[i] for i in dev_indices]
    dev_labels = [labels[i] for i in dev_indices]

    # Try Different Hyperparameter Combinations
    for k_rarest in k_rarest_values:
        for m in m_values:
            for n_most in n_most_values:
                # Build Vocabulary from Training Data
                print("Building vocabulary...")
                vocab = build_vocabulary(train_texts, train_labels, n_most=n_most, k_rarest=k_rarest, m=m)

                # Vectorize Texts
                print("Vectorizing texts...")
                X_train = vectorize_texts(train_texts, vocab)
                X_dev = vectorize_texts(dev_texts, vocab)
                y_train = np.array(train_labels)
                y_dev = np.array(dev_labels)

                for n_estimators in n_estimators_values:
                    print(f"\n--- Training with n_estimators={n_estimators}, m={m}, n_most={n_most}, k_rarest={k_rarest} ---")

                    # Convert to PyTorch Tensors
                    X_train = to_tensor(X_train)
                    y_train = to_tensor(y_train)
                    X_dev = to_tensor(X_dev)
                    y_dev = to_tensor(y_dev)

                    # Train Random Forest
                    print(f"Training Random Forest with {n_estimators} trees...")
                    forest = randomforest_train(X_train.cpu().numpy(), y_train.cpu().numpy(), n_estimators)

                    # Evaluate on the Development Set
                    dev_preds = randomforest_predict(X_dev.cpu().numpy(), forest)
                    dev_preds_torch = torch.tensor(dev_preds, device="cuda")
                    dev_acc = torch.mean((dev_preds_torch == y_dev).float())
                    print(f"Development Accuracy: {dev_acc * 100:.2f}%")

                    # Train Sklearn Random Forest
                    print("\nTraining Sklearn Random Forest classifier...")
                    sklearn_model = RandomForestClassifier(
                        n_estimators=n_estimators, 
                        max_features='sqrt',
                        criterion="entropy",  # ID3 uses entropy as criterion
                        bootstrap=True,  # Use Bagging
                        random_state=SEED
                    )
                    sklearn_model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())

                    # Evaluate on the Development Set
                    sklearn_dev_preds = sklearn_model.predict(X_dev.cpu().numpy())
                    sklearn_dev_acc = np.mean(sklearn_dev_preds == y_dev.cpu().numpy())
                    print(f"Sklearn Random Forest Dev Accuracy: {sklearn_dev_acc * 100:.2f}%")

                    # Save the best model
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        best_params = {'n_estimators': n_estimators, 'm': m, 'n_most': n_most, 'k_rarest': k_rarest}

                        # Save Sklearn Random Forest Model
                        results_dir = os.path.join('..', 'results')
                        sklearn_model_path = os.path.join(results_dir, 'sklearn_randomforest.pkl')
                        with open(sklearn_model_path, 'wb') as f:
                            pickle.dump(sklearn_model, f)
                        print(f"Sklearn Random Forest model saved to {sklearn_model_path}")

                        # Save the Model and Vocabulary
                        results_dir = os.path.join('..', 'results')
                        os.makedirs(results_dir, exist_ok=True)
                        model_path = os.path.join(results_dir, 'randomforest_model.pkl')
                        vocab_path = os.path.join(results_dir, 'vocab_RF.pkl')
                        with open(model_path, 'wb') as f:
                            pickle.dump(forest, f)
                        with open(vocab_path, 'wb') as f:
                            pickle.dump(vocab, f)
                        print(f"Model and vocabulary saved to {model_path} and {vocab_path}")

                        # Learning Curve Experiment
                        print("\nRunning learning curve experiment (evaluating for positive class)...")
                        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
                        train_sizes, train_prec, train_rec, train_f1 = [], [], [], []
                        dev_prec, dev_rec, dev_f1 = [], [], []
                        sklearn_train_f1, sklearn_dev_f1 = [], []

                        # Print header for Part A
                        # print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Size", "Train Prec", "Train Rec", "Train F1", "Dev Prec", "Dev Rec", "Dev F1"))

                        # Print header for Part B
                        print("{:<10} {:<20} {:<20} {:<20} {:<20}".format("Size", "Custom Train F1", "Custom Dev F1", "Sklearn Train F1", "Sklearn Dev F1"))

                        # For reproducibility, we use the first N examples of the (already shuffled) training data.
                        for frac in fractions:
                            subset_size = int(frac * len(train_texts))
                            sub_train_texts = train_texts[:subset_size]
                            sub_train_labels = to_tensor(train_labels[:subset_size])
                            X_sub_train = vectorize_texts(sub_train_texts, vocab)

                            # Train Random Forest on the subset
                            forest_subset = randomforest_train(X_sub_train, sub_train_labels.cpu().numpy(), n_estimators)

                            # Evaluate on Training Subset and Full Development Set
                            train_preds_subset = randomforest_predict(X_sub_train, forest_subset)
                            dev_preds_subset = randomforest_predict(X_dev.cpu().numpy(), forest_subset)

                            prec_train, rec_train, f1_train = compute_metrics_for_class_torch(sub_train_labels, torch.tensor(train_preds_subset, device="cuda"), target=1)
                            prec_dev, rec_dev, f1_dev = compute_metrics_for_class_torch(y_dev, torch.tensor(dev_preds_subset, device="cuda"), target=1)

                            # Sklearn Random Forest on subset
                            sklearn_model_subset = RandomForestClassifier(
                                n_estimators=n_estimators, 
                                max_features='sqrt',
                                criterion="entropy",  # ID3 uses entropy as splitting criterion
                                bootstrap=True,  # Ensure Bagging is used
                                random_state=SEED
                            )
                            sklearn_model_subset.fit(X_sub_train.cpu().numpy(), sub_train_labels.cpu().numpy())
                            sklearn_train_preds = sklearn_model_subset.predict(X_sub_train.cpu().numpy())
                            sklearn_dev_preds_subset = sklearn_model_subset.predict(X_dev.cpu().numpy())
                            prec_train_sklearn, rec_train_sklearn, f1_train_sklearn = compute_metrics_for_class_sklearn(sub_train_labels.cpu().numpy(), sklearn_train_preds, target=1)
                            prec_dev_sklearn, rec_dev_sklearn, f1_dev_sklearn = compute_metrics_for_class_sklearn(y_dev.cpu().numpy(), sklearn_dev_preds_subset, target=1)

                            # Print results for Part A
                            # print("{:<10} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(subset_size, prec_train, rec_train, f1_train, prec_dev, rec_dev, f1_dev))

                            # Print results for Part B
                            print("{:<10} {:<20.4f} {:<20.4f} {:<20.4f} {:<20.4f}".format(subset_size, f1_train, f1_dev, f1_train_sklearn, f1_dev_sklearn))

                            train_sizes.append(subset_size)
                            train_prec.append(prec_train)
                            train_rec.append(rec_train)
                            train_f1.append(f1_train)
                            dev_prec.append(prec_dev)
                            dev_rec.append(rec_dev)
                            dev_f1.append(f1_dev)
                            sklearn_train_f1.append(f1_train_sklearn)
                            sklearn_dev_f1.append(f1_dev_sklearn)

                        # Plot learning curve
                        plot_learning_curve_A(train_sizes, train_prec, train_rec, train_f1, dev_prec, dev_rec, dev_f1)
                        plot_learning_curve_B(train_sizes, train_f1, dev_f1, sklearn_train_f1, sklearn_dev_f1)

    print("\n--- Best Hyperparameters ---")
    print(best_params)

    # End time
    end_time = datetime.datetime.now()
    print("Duration:",end_time-start_time)
    print("\nEnd time:", end_time)


if __name__ == "__main__":
    main()
