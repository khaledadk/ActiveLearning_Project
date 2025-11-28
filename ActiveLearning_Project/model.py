import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class ActiveLearningModel:
    def __init__(self, model_type='svm'):
        self.model_type = model_type
        if model_type == 'svm':
            self.model = SVC(probability=True, kernel='rbf', random_state=42)
        
    def train(self, X_labeled, y_labeled):
        """Entraîne le modèle sur les données annotées"""
        self.model.fit(X_labeled, y_labeled)
        
    def predict(self, X):
        """Prédictions sur de nouvelles données"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Probabilités des prédictions (pour l'incertitude)"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Évalue le modèle et retourne l'accuracy"""
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

class ActiveLearningExperiment:
    def __init__(self, X_pool, y_pool, X_test, y_test, initial_size=10):
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.X_test = X_test
        self.y_test = y_test
        self.initial_size = initial_size
        
    def uncertainty_sampling(self, model, X_unlabeled, batch_size=10):
        """Stratégie d'échantillonnage par incertitude (entropie)"""
        probabilities = model.predict_proba(X_unlabeled)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        selected_indices = np.argsort(entropy)[-batch_size:]
        return selected_indices
    
    def random_sampling(self, X_unlabeled, batch_size=10):
        """Stratégie d'échantillonnage aléatoire"""
        n_samples = len(X_unlabeled)
        batch_size = min(batch_size, n_samples)
        return np.random.choice(n_samples, size=batch_size, replace=False)
    
    def run_experiment(self, strategy='uncertainty', n_iterations=50, batch_size=10, model_type='svm'):
        """Exécute une expérience d'Active Learning"""
        np.random.seed(42)
        initial_indices = np.random.choice(len(self.X_pool), size=self.initial_size, replace=False)
        
        labeled_indices = list(initial_indices)
        unlabeled_indices = [i for i in range(len(self.X_pool)) if i not in labeled_indices]
        
        accuracies = []
        model = ActiveLearningModel(model_type)
        
        print(f"🚀 Début de l'expérience - Stratégie: {strategy}")
        print(f"Échantillons initiaux: {len(labeled_indices)}")
        print(f"Échantillons non annotés: {len(unlabeled_indices)}")
        
        for iteration in range(n_iterations):
            X_labeled = self.X_pool[labeled_indices]
            y_labeled = self.y_pool[labeled_indices]
            model.train(X_labeled, y_labeled)
            
            acc = model.evaluate(self.X_test, self.y_test)
            accuracies.append(acc)
            
            if not unlabeled_indices:
                print("⚠️ Plus d'échantillons non annotés!")
                break
            
            X_unlabeled = self.X_pool[unlabeled_indices]
            
            if strategy == 'uncertainty':
                selected = self.uncertainty_sampling(model, X_unlabeled, batch_size)
            elif strategy == 'least_confidence':
                selected = self.least_confidence_sampling(model, X_unlabeled, batch_size)
            elif strategy == 'margin':
                selected = self.margin_sampling(model, X_unlabeled, batch_size)
            else:
                 selected = self.random_sampling(X_unlabeled, batch_size)

            
            new_labeled_indices = [unlabeled_indices[i] for i in selected]
            labeled_indices.extend(new_labeled_indices)
            
            for idx in new_labeled_indices:
                unlabeled_indices.remove(idx)
            
            if (iteration + 1) % 5 == 0:
                print(f"📊 {strategy} - Itération {iteration+1}: Accuracy = {acc:.4f}, "
                      f"Échantillons annotés: {len(labeled_indices)}")
        
        final_acc = model.evaluate(self.X_test, self.y_test)
        print(f"✅ {strategy} - Accuracy finale: {final_acc:.4f}")
        print(f"📈 Échantillons annotés totaux: {len(labeled_indices)}\n")
        
        return accuracies, len(labeled_indices)

    def least_confidence_sampling(self, model, X_unlabeled, batch_size=10):
        """Stratégie de type Least Confidence"""
        probabilities = model.predict_proba(X_unlabeled)
        # score = 1 - probabilité maximale
        uncertainty = 1 - np.max(probabilities, axis=1)
        selected_indices = np.argsort(uncertainty)[-batch_size:]
        return selected_indices

    def margin_sampling(self, model, X_unlabeled, batch_size=10):
        """Stratégie de type Margin Sampling"""
        probabilities = model.predict_proba(X_unlabeled)
        # On trie par probas décroissantes pour obtenir p1 et p2
        sorted_proba = np.sort(probabilities, axis=1)
        p1 = sorted_proba[:, -1]   # meilleure classe
        p2 = sorted_proba[:, -2]   # deuxième meilleure classe
        margin = p1 - p2           # plus petit = plus ambigu
        selected_indices = np.argsort(margin)[:batch_size]
        return selected_indices
