import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)


np.random.seed(42)
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------------------
# STEP 2: Load the Covertype dataset (581,012 samples, 54 features, 7 classes)
# ------------------------------------------------------------------------------
print("Loading Covertype dataset from UCI repository via Scikit-Learn...")
covtype_data = fetch_covtype(as_frame=True)
X_raw = covtype_data.data
y_raw = covtype_data.target

print(f"Dataset Loaded: {X_raw.shape[0]:,} samples, {X_raw.shape[1]} features")


# ------------------------------------------------------------------------------
# STEP 3-5: Class Distribution Analysis
# ------------------------------------------------------------------------------
class_names = [
    "Spruce/Fir",          # Class 1
    "Lodgepole Pine",      # Class 2
    "Ponderosa Pine",      # Class 3
    "Cottonwood/Willow",   # Class 4
    "Aspen",               # Class 5
    "Douglas-fir",         # Class 6
    "Krummholz"            # Class 7
]

class_counts = pd.Series(y_raw).value_counts().sort_index()
max_count = class_counts.max()
min_count = class_counts.min()
imbalance_ratio = max_count / min_count

print("\n--- Class Distribution Analysis ---")
for cls_id, count in class_counts.items():
    pct = (count / len(y_raw)) * 100
    print(f"Class {cls_id} ({class_names[cls_id-1]}): {count:,} samples ({pct:.2f}%)")

print(f"\nImbalance Ratio (Most Frequent / Least Frequent): {imbalance_ratio:.2f}:1")


# ------------------------------------------------------------------------------
# STEP 6-8: Prepare Features and Target
# ------------------------------------------------------------------------------
X_features = X_raw.values
y_zero_indexed = y_raw.values - 1  # Map 1..7 to 0..6

def one_hot_encode(y_labels, K=7):
    """Encodes integer target labels into binary matrix [N x K]."""
    return np.eye(K)[y_labels]

Y_onehot_full = one_hot_encode(y_zero_indexed, K=7)
print(f"One-Hot Target Matrix Shape: {Y_onehot_full.shape}")


# ------------------------------------------------------------------------------
# STEP 9-10: Partition Dataset and Preprocess
# ------------------------------------------------------------------------------
X_train_raw, X_temp_raw, y_train_labels, y_temp_labels = train_test_split(
    X_features, y_zero_indexed, test_size=0.20, random_state=42, stratify=y_zero_indexed
)

X_val_raw, X_test_raw, y_val_labels, y_test_labels = train_test_split(
    X_temp_raw, y_temp_labels, test_size=0.50, random_state=42, stratify=y_temp_labels
)

print(f"\nData Partitioning:")
print(f"  Training Set  : {X_train_raw.shape[0]:,} samples (80%)")
print(f"  Validation Set: {X_val_raw.shape[0]:,} samples (10%)")
print(f"  Testing Set   : {X_test_raw.shape[0]:,} samples (10%)")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

N_train = len(y_train_labels)
K_classes = 7
counts_per_class = np.bincount(y_train_labels, minlength=K_classes)
class_weights = N_train / (K_classes * counts_per_class)

print("\nComputed Class Weights (w_c):")
for c in range(K_classes):
    print(f"  Class {c} ({class_names[c]}): w_{c} = {class_weights[c]:.4f}")

Y_train_oh = one_hot_encode(y_train_labels, K_classes)
Y_val_oh = one_hot_encode(y_val_labels, K_classes)
Y_test_oh = one_hot_encode(y_test_labels, K_classes)


# ------------------------------------------------------------------------------
# Deep Feed-Forward Neural Network Class
# ------------------------------------------------------------------------------
class DeepFeedForwardNN:
    """
    Deep Feed-Forward Neural Network (DFNN) for 7-class Covertype Classification.
    Architecture: [54, 64, 32, 16, 7]
    Hidden Activations: ReLU
    Output Activation: Softmax
    Loss Function: Weighted Categorical Cross-Entropy
    """
    def __init__(self, layer_dims=[54, 64, 32, 16, 7]):
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1 
        self.params = {}
        for l in range(1, self.L + 1):
            n_in = layer_dims[l - 1]
            n_out = layer_dims[l]
            self.params[f'W{l}'] = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            self.params[f'b{l}'] = np.zeros((1, n_out))

    @staticmethod
    def relu(Z):
        """ReLU Activation: f(z) = max(0, z)"""
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        """ReLU Derivative: f'(z) = 1 if z > 0 else 0"""
        return (Z > 0).astype(float)

    @staticmethod
    def softmax(Z):
        """Softmax Activation: \sigma(z)_k = e^{z_k} / \sum e^{z_j}"""  # FIX 1: Removed dangling 'y'
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        """Vectorized Forward Propagation"""
        cache = {'A0': X}
        A = X
        for l in range(1, self.L):
            Z = np.dot(A, self.params[f'W{l}']) + self.params[f'b{l}']
            A = self.relu(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        ZL = np.dot(A, self.params[f'W{self.L}']) + self.params[f'b{self.L}']
        AL = self.softmax(ZL)
        cache[f'Z{self.L}'] = ZL
        cache[f'A{self.L}'] = AL
        
        return AL, cache

    def compute_weighted_loss(self, Y_hat, Y_onehot, class_weights):
        """
        Computes Weighted Categorical Cross-Entropy Loss
        """
        sample_weights = np.sum(Y_onehot * class_weights, axis=1, keepdims=True)
        eps = 1e-15 
        Y_hat_clipped = np.clip(Y_hat, eps, 1.0 - eps)
        
        ce_loss = -np.sum(Y_onehot * np.log(Y_hat_clipped), axis=1, keepdims=True)
        weighted_loss = np.mean(sample_weights * ce_loss)
        return weighted_loss

    def backward(self, cache, Y_onehot, class_weights):
        """Vectorized Backpropagation with Class Weights"""
        grads = {}
        B = Y_onehot.shape[0]
        AL = cache[f'A{self.L}']
        sample_weights = np.sum(Y_onehot * class_weights, axis=1, keepdims=True)
        dZ = (sample_weights * (AL - Y_onehot)) / B
        for l in reversed(range(1, self.L + 1)):
            A_prev = cache[f'A{l-1}']
            grads[f'dW{l}'] = np.dot(A_prev.T, dZ)
            grads[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True)
            if l > 1:
                dA_prev = np.dot(dZ, self.params[f'W{l}'].T)
                dZ = dA_prev * self.relu_derivative(cache[f'Z{l-1}'])
        return grads

    def update_parameters(self, grads, learning_rate):
        """Parameter Update via Mini-batch SGD"""
        for l in range(1, self.L + 1):
            self.params[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.params[f'b{l}'] -= learning_rate * grads[f'db{l}']


# ------------------------------------------------------------------------------
# Model Training
# ------------------------------------------------------------------------------
model = DeepFeedForwardNN(layer_dims=[54, 64, 32, 16, 7])
BATCH_SIZE = 128
LEARNING_RATE = 0.05
MAX_EPOCHS = 100
PATIENCE = 15

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = float('inf')
best_weights = None
patience_counter = 0

print(f"\n--- Training DFNN Model (Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Patience: {PATIENCE}) ---")

for epoch in range(1, MAX_EPOCHS + 1):
    shuffled_indices = np.random.permutation(N_train)
    X_shuffled = X_train[shuffled_indices]
    Y_shuffled = Y_train_oh[shuffled_indices]
    
    # FIX 2: Compute running batch metrics during training loop to speed up epoch evaluation
    running_loss = 0.0
    correct_train = 0
    
    for i in range(0, N_train, BATCH_SIZE):
        X_batch = X_shuffled[i : i + BATCH_SIZE]
        Y_batch = Y_shuffled[i : i + BATCH_SIZE]
        
        Y_hat_batch, cache = model.forward(X_batch)
        grads = model.backward(cache, Y_batch, class_weights)
        model.update_parameters(grads, LEARNING_RATE)
        
        batch_loss = model.compute_weighted_loss(Y_hat_batch, Y_batch, class_weights)
        running_loss += batch_loss * X_batch.shape[0]
        correct_train += np.sum(np.argmax(Y_hat_batch, axis=1) == np.argmax(Y_batch, axis=1))

    tr_loss = running_loss / N_train
    tr_acc = correct_train / N_train
    
    Y_val_pred_prob, _ = model.forward(X_val)
    v_loss = model.compute_weighted_loss(Y_val_pred_prob, Y_val_oh, class_weights)
    v_acc = accuracy_score(y_val_labels, np.argmax(Y_val_pred_prob, axis=1))
    
    train_losses.append(tr_loss)
    val_losses.append(v_loss)
    train_accuracies.append(tr_acc)
    val_accuracies.append(v_acc)
    
    print(f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} | Train Loss: {tr_loss:.4f}, Val Loss: {v_loss:.4f} | Train Acc: {tr_acc:.4f}, Val Acc: {v_acc:.4f}")
    
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        best_weights = {k: v.copy() for k, v in model.params.items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n[Early Stopping Triggered] Validation loss failed to improve for {PATIENCE} epochs.")
            print(f"Restoring best model parameters from Epoch {epoch - PATIENCE}...")
            model.params = best_weights
            break


# ------------------------------------------------------------------------------
# Test Set Evaluation & Visualization
# ------------------------------------------------------------------------------
Y_test_probs, _ = model.forward(X_test)
y_test_pred_labels = np.argmax(Y_test_probs, axis=1)
test_acc = accuracy_score(y_test_labels, y_test_pred_labels)

p_per_cls, r_per_cls, f1_per_cls, _ = precision_recall_fscore_support(
    y_test_labels, y_test_pred_labels, average=None, zero_division=0
)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_test_labels, y_test_pred_labels, average='macro', zero_division=0
)
cm_matrix = confusion_matrix(y_test_labels, y_test_pred_labels)

print("\n==============================================================================")
print("                           TEST SET EVALUATION RESULTS                        ")
print("==============================================================================")
print(f"Overall Test Accuracy : {test_acc * 100:.2f}%")
print(f"Macro Average Precision: {macro_p:.4f}")
print(f"Macro Average Recall   : {macro_r:.4f}")
print(f"Macro Average F1-Score : {macro_f1:.4f}\n")

print("Detailed Classification Report:")
print(classification_report(y_test_labels, y_test_pred_labels, target_names=class_names, digits=4))


plt.figure(figsize=(18, 16))

plt.subplot(3, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Weighted Loss', color='#1f77b4', linewidth=2)
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Weighted Loss', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('Step 24: Training & Validation Loss', fontweight='bold', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='#2ca02c', linewidth=2)
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='#d62728', linestyle='--', linewidth=2)
plt.title('Step 24: Training & Validation Accuracy', fontweight='bold', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 3)
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.title('Step 25: 7x7 Confusion Matrix Heatmap', fontweight='bold', fontsize=12)
plt.xlabel('Predicted Cover Type')
plt.ylabel('True Cover Type')
plt.xticks(rotation=35, ha='right')

plt.subplot(3, 2, 4)
bars = plt.bar(class_names, f1_per_cls, color='#9467bd', alpha=0.85, edgecolor='black')
plt.title('Step 26: Per-Class F1 Scores', fontweight='bold', fontsize=12)
plt.ylabel('F1-Score')
plt.ylim([0, 1.05])
plt.xticks(rotation=35, ha='right')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

plt.subplot(3, 2, 5)
x_indices = np.arange(K_classes)
bar_width = 0.35
plt.bar(x_indices - bar_width/2, p_per_cls, bar_width, label='Precision', color='#1f77b4')
plt.bar(x_indices + bar_width/2, r_per_cls, bar_width, label='Recall', color='#ff7f0e')
plt.title('Step 27: Precision vs Recall Trade-off Per Class', fontweight='bold', fontsize=12)
plt.xticks(x_indices, class_names, rotation=35, ha='right')
plt.ylabel('Score')
plt.ylim([0, 1.05])
plt.legend()

plt.subplot(3, 2, 6)
true_counts = np.bincount(y_test_labels, minlength=K_classes)
pred_counts = np.bincount(y_test_pred_labels, minlength=K_classes)
plt.bar(x_indices - bar_width/2, true_counts, bar_width, label='True Test Distribution', color='#2ca02c')
plt.bar(x_indices + bar_width/2, pred_counts, bar_width, label='Predicted Distribution', color='#d62728')
plt.title('Step 28: True vs Predicted Class Distributions', fontweight='bold', fontsize=12)
plt.xticks(x_indices, class_names, rotation=35, ha='right')
plt.ylabel('Sample Count')
plt.legend()

plt.tight_layout()
plt.show()