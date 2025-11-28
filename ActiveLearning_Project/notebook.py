# 🎯 Projet Active Learning - Reconnaissance de Caractères Arabes
# Master 2 Sciences de Données et Analyses

# 📦 IMPORTATION DES LIBRAIRIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

plt.style.use('default')
sns.set_palette("husl")
np.random.seed(42)

from model import ActiveLearningExperiment

print("✅ Toutes les librairies sont importées!")

# 📥 CHARGEMENT DES DONNÉES
df = pd.read_csv("dataset/dataset.txt", header=None, sep=",")
feature_names = [f"f{i}" for i in range(df.shape[1]-1)] + ["label"]
df.columns = feature_names

print("📊 Shape du dataset:", df.shape)
df.head()

# 🔍 EXPLORATION DES DONNÉES
print(f"Nombre total d'échantillons: {df.shape[0]}")
print(f"Nombre de features: {df.shape[1] - 1}")

target_counts = df['label'].value_counts()
print(f"Nombre de classes: {len(target_counts)}")
print(target_counts)

plt.figure(figsize=(12, 6))
target_counts.plot(kind='bar')
plt.title('Distribution des Caractères Arabes')
plt.xlabel('Caractères')
plt.ylabel("Nombre d'échantillons")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🔧 PRÉTRAITEMENT DES DONNÉES
X = df.iloc[:, :-1].to_numpy()
y = df['label'].to_numpy()

# Encodage
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 📊 BASELINE
np.random.seed(42)
initial_indices = np.random.choice(len(X_train), size=10, replace=False)
X_baseline = X_train[initial_indices]
y_baseline = y_train[initial_indices]

baseline_model = SVC(probability=True, random_state=42)
baseline_model.fit(X_baseline, y_baseline)
y_pred_baseline = baseline_model.predict(X_test)
baseline_accuracy = np.mean(y_pred_baseline == y_test)

full_model = SVC(probability=True, random_state=42)
full_model.fit(X_train, y_train)
y_pred_full = full_model.predict(X_test)
full_accuracy = np.mean(y_pred_full == y_test)

print(f"Baseline Accuracy (10 échantillons): {baseline_accuracy:.4f}")
print(f"Full Data Accuracy: {full_accuracy:.4f}")
print(f"Gain potentiel: {full_accuracy - baseline_accuracy:.4f}")

# 🎯 ACTIVE LEARNING
X_pool = X_train
y_pool = y_train
experiment = ActiveLearningExperiment(X_pool, y_pool, X_test, y_test, initial_size=10)

# Random Sampling
random_acc, _ = experiment.run_experiment(strategy='random', n_iterations=10, batch_size=5)
# Uncertainty Sampling
uncertainty_acc, _ = experiment.run_experiment(strategy='uncertainty', n_iterations=10, batch_size=5)

# 📊 VISUALISATION
plt.figure(figsize=(10,6))
iterations = range(1, len(random_acc)+1)
plt.plot(iterations, random_acc, label='Random Sampling', marker='o', linewidth=2)
plt.plot(iterations, uncertainty_acc, label='Uncertainty Sampling', marker='x', linewidth=2)
plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label=f'Baseline (10 samples)')
plt.axhline(y=full_accuracy, color='g', linestyle='--', label=f'Full Data ({len(X_train)} samples)')
plt.xlabel('Itérations')
plt.ylabel('Accuracy')
plt.title('Active Learning - Comparaison des stratégies')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
