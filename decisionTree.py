import numpy as np
import pickle
import pandas as pd
import h5py
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
import multiprocessing
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np


def gini_impurity(y):
    """Calcular la impureza Gini de un array de etiquetas"""
    if len(y) == 0:
        return 0
    # Proporción de cada clase
    clases = np.unique(y)
    probabilities = [np.mean(y == c) for c in clases]
    gini = 1 - sum(p ** 2 for p in probabilities)
    return gini

def entropy(y):
    """Calcular la entropía de un array de etiquetas"""
    if len(y) == 0:
        return 0
    # Proporción de cada clase
    clases = np.unique(y)
    probabilities = [np.mean(y == c) for c in clases]
    # Usamos log2 para calcular la entropía
    entropy_value = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy_value


def information_gain(y, left_indices, right_indices, impurity_function=gini_impurity):
    """Calcular la ganancia de información de una división utilizando impureza Gini o entropía"""
    parent_impurity = impurity_function(y)
    
    # Subconjuntos izquierdo y derecho
    left_impurity = impurity_function(y[left_indices])
    right_impurity = impurity_function(y[right_indices])
    
    # Peso de los subconjuntos izquierdo y derecho
    left_weight = len(left_indices) / len(y)
    right_weight = len(right_indices) / len(y)
    
    # Impureza ponderada de los hijos
    weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
    
    # Ganancia de información
    info_gain = parent_impurity - weighted_impurity
    return info_gain


def calculate_clusters_based_on_variance(column):
    variance = np.var(column)
    
    if variance < 1:
        return 2 
    elif variance < 10:
        return 3  
    else:
        return 4 
    
def discretize_with_kmeans(column):
    n_clusters = calculate_clusters_based_on_variance(column)
    
    # Reshape para aplicar KMeans (n_samples, n_features)
    column_reshaped = column.values.reshape(-1, 1)
    
    # Ajustamos KMeans a la columna con el número de clusters calculado
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(column_reshaped)
    
    # Asignar a cada valor el índice del cluster
    return kmeans.labels_

def discretize_all(dataset):
    x_df_discretized = dataset

    for col in dataset.columns:
        n_clusters = 3  
        x_df_discretized[col] = discretize_with_kmeans(dataset[col])
    return x_df_discretized


class DecisionTree:
    """Decision Tree tomando en cuenta que el dataset es discreto o está discretizado"""
    def __init__(self, max_depth=None, impurity_function=gini_impurity):
        self.max_depth = max_depth
        self.impurity_function = impurity_function
        self.tree = None
    def fit(self, X, y):
        """Ajustar el árbol de decisión a los datos"""
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Si ya es un nodo hoja o alcanzamos la profundidad máxima ya no dividimos
        if n_labels == 1 or (self.max_depth and depth == self.max_depth):
            return np.unique(y)[0]

        # Buscamos el atributo que gane más información
        best_gain = -1 # Valor de inicio
        best_split = None
        best_left_indices = None
        best_right_indices = None

        # Por cada caracterísitca o dimensión
        for feature in range(n_features):
            valores = np.unique(X[:, feature]) 
            # Sacamos las ramas
            for valor in valores: 
                left_indices = np.where(X[:, feature] == valor)[0]
                right_indices = np.where(X[:, feature] != valor)[0]

                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                gain = information_gain(y, left_indices, right_indices, self.impurity_function)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature": feature,
                        "value": valor 
                    }
                    best_left_indices = left_indices
                    best_right_indices = right_indices
             # Si no se encontró una mejor división, retornamos la clase mayoritaria
        if best_gain == -1:
            return np.bincount(y).argmax()
            
        # Construir recursivamente las ramas
        left_subtree = self._build_tree(X[best_left_indices], y[best_left_indices], depth + 1)
        right_subtree = self._build_tree(X[best_right_indices], y[best_right_indices], depth + 1)
        return {
        "feature": best_split["feature"],
        "value": best_split["value"],
        "left": left_subtree,
        "right": right_subtree
        }
        
    def predict_one(self, x, node=None):
        """Predecir una muestra individual"""
        if node is None:
            node = self.tree

        if not isinstance(node, dict):
            return node

        feature = node["feature"]
        value = node["value"]

        if x[feature] == value:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])
    
    def predict(self, X):
        """Predecir múltiples muestras"""
        return np.array([self.predict_one(x) for x in X])

    def print_tree(self, node=None, depth=0):
            """Imprimir el árbol de decisión"""
            if node is None:
                node = self.tree
            indent = "  " * depth
            if not isinstance(node, dict):
                print(indent + "Predict:", node)
                return
        
            feature = node["feature"]
            value = node["value"]
        
            print(f"{indent}¿X[{feature}] == {value}?")
            print(f"{indent}-> Sí:")
            self.print_tree(node["left"], depth + 1)
        
            print(f"{indent}-> No:")
            self.print_tree(node["right"], depth + 1)

class DecisionTreeEnsemble:
    def __init__(self, n_estimators=3, max_depth=None, impurity_function=gini_impurity):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.impurity_function = impurity_function
        self.trees = []
        self.accuracies = []
    def _bootstrap_sample(self, X, y):
        """Tomar una muestra con reemplazo del conjunto de entrenamiento"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def calculate_where_fail(self, model, X, y):
        """Retornar índices donde el modelo falló"""
        preds = model.predict(X)
        return np.where(preds != y)[0]


    def increment_weight(self, X, y, fail_indices, l=2):
        """Replicar l veces las muestras mal clasificadas"""
        X_fail = X[fail_indices]
        y_fail = y[fail_indices]
        X_augmented = np.concatenate([X] + [X_fail] * l)
        y_augmented = np.concatenate([y] + [y_fail] * l)
        return X_augmented, y_augmented

    def fit(self, X, y):
        self.trees = []
        self.accuracies = [] 
        X_current = X
        y_current = y

        for i in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, impurity_function=self.impurity_function)
            X_sample, y_sample = self._bootstrap_sample(X_current, y_current)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

            # identificar fallos y reponderar para la próxima iteración
            fail_indices = self.calculate_where_fail(tree, X, y)
            if len(fail_indices) > 0:
                X_current, y_current = self.increment_weight(X, y, fail_indices, l=2)  
            y_pred = tree.predict(X)
            acc = accuracy_score(y, y_pred)
            self.accuracies.append(acc)
            print(f"[Árbol {i+1}] Accuracy en entrenamiento: {acc:.4f}")
            print(classification_report(y, y_pred))

    def predict(self, X):
        # Cada árbol hace su predicción
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Votación por mayoría
        final_preds = []
        for sample_preds in tree_preds.T:  
            majority_vote = Counter(sample_preds).most_common(1)[0][0]
            final_preds.append(majority_vote)
        return np.array(final_preds)
    def best_tree(self):
        """Retorna el árbol con el mayor accuracy"""
        best_accuracy_index = np.argmax(self.accuracies)  
        best_tree = self.trees[best_accuracy_index]  
        best_accuracy = self.accuracies[best_accuracy_index] 
        print(f"El mejor árbol tiene un accuracy de: {best_accuracy:.4f}")
        return best_tree
    def saveBestTree(self, filename="best_tree.pkl"):
        """Guarda el árbol con el mejor accuracy en un archivo usando pickle sin numpy para ya no volver a correr"""
        best_tree = self.best_tree()  
        with open(filename, 'wb') as f:
            pickle.dump(best_tree, f) 
        print(f"El mejor árbol ha sido guardado en {filename}")


